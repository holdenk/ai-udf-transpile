package ai_udf;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.sql.api.java.UDF1;

public class FhiDeidentifyTextTuned implements UDF1<Object, Object> {

    // One pattern set per input domain. For pure-ASCII text java.util.regex's
    // default classes mean what Python's unicode-aware ones mean, so the
    // cheap spellings are exact there; anything else takes the faithful set.
    private static final class Pats {
        final Pattern PLACEHOLDER, SSN, PHONE, EMAIL, DOB, MEMBER, ADDRESS, RESTORE;

        Pats(boolean ascii) {
            final String D = ascii ? "[0-9]" : "\\p{Nd}";
            final String SC = ascii
                ? "\\t\\n\\x0B\\f\\r\\x1C-\\x1F "
                : "\\t\\n\\x0B\\f\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000";
            final String S = "[" + SC + "]";
            final String WC = ascii ? "A-Za-z0-9_" : "\\p{L}\\p{N}_";
            final String B = ascii
                ? "\\b"
                : "(?:(?<=[" + WC + "])(?![" + WC + "])|(?<![" + WC + "])(?=[" + WC + "]))";
            final String L = ascii ? "[A-Za-z]" : "[A-Za-z\\u0130\\u0131\\u017F\\u212A]";
            final String MM = "(?:0[1-9]|1[0-2])";
            final String DD = "(?:0[1-9]|[12]" + D + "|3[01])";
            final String YY = "(?:19|20)" + D + "{2}";
            PLACEHOLDER = Pattern.compile(
                "\\{\\{(?:FIRST_NAME|LAST_NAME|NAME|PHONE|Your Phone Number|SSN|EMAIL|Your Email Address|DOB|ADDRESS|MEMBER_ID|PLAN_ID|CASEID|SCSID|GPID|FAX|DATE|ID|REDACTED)[^}]*\\}\\}"
                + "|\\[(?:NAME|PHONE|SSN|EMAIL|DOB|ADDRESS|MEMBER_ID|PLAN_ID|FAX|DATE|ID|REDACTED)[^\\]]*\\]");
            SSN = Pattern.compile(B + D + "{3}-" + D + "{2}-" + D + "{4}" + B);
            PHONE = Pattern.compile(
                "(?<![" + D.replace("[", "").replace("]", "") + "-])(?:\\+?1[-." + SC + "]?)?\\(?" + D + "{3}\\)?[-." + SC + "]?"
                + D + "{3}[-." + SC + "]?" + D + "{4}(?![" + D.replace("[", "").replace("]", "") + "-])");
            EMAIL = Pattern.compile(B + "[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}" + B);
            DOB = Pattern.compile(
                "(?:" + ci(ascii, "DOB")
                + "|" + ci(ascii, "date") + S + "+" + ci(ascii, "of") + S + "+" + ci(ascii, "birth")
                + "|" + ci(ascii, "born")
                + "|" + ci(ascii, "birthday")
                + "|" + ci(ascii, "d") + "\\." + ci(ascii, "o") + "\\." + ci(ascii, "b") + "\\.?)"
                + S + "*:?" + S + "*"
                + "(?:" + MM + "[/\\-]" + DD + "[/\\-]" + YY
                + "|" + YY + "[/\\-]" + MM + "[/\\-]" + DD + ")");
            MEMBER = Pattern.compile(
                alt(ascii, new String[] {"member", "group", "subscriber", "policy", "plan"})
                + S + "*"
                + "(?:" + ci(ascii, "id") + "|#|" + ci(ascii, "number") + "|" + ci(ascii, "no") + "\\.?)"
                + S + "*:?" + S + "*" + L + "{0,4}" + D + "{6,12}" + B);
            ADDRESS = Pattern.compile(
                B + D + "{1,5}" + S + "+(?:" + L + L + "+" + S + "*){1,3}"
                + alt(ascii, new String[] {"St", "Street", "Ave", "Avenue", "Blvd", "Boulevard", "Dr", "Drive",
                                           "Ln", "Lane", "Rd", "Road", "Ct", "Court", "Way", "Pl", "Place",
                                           "Cir", "Circle"})
                + "\\.?" + B);
            RESTORE = Pattern.compile("\\x00PH(" + D + "+)\\x00");
        }
    }

    private static final Pats ASCII = new Pats(true);
    private static final Pats FULL = new Pats(false);

    // Python (?i) over ASCII letters: i, s and k also match their non-ASCII
    // case partners, which cannot occur in pure-ASCII input.
    private static String ci(boolean ascii, String word) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < word.length(); i++) {
            char c = Character.toLowerCase(word.charAt(i));
            char u = Character.toUpperCase(c);
            sb.append('[').append(c).append(u);
            if (!ascii) {
                if (c == 'i') {
                    sb.append("\\u0130\\u0131");
                } else if (c == 's') {
                    sb.append("\\u017F");
                } else if (c == 'k') {
                    sb.append("\\u212A");
                }
            }
            sb.append(']');
        }
        return sb.toString();
    }

    private static String alt(boolean ascii, String[] words) {
        StringBuilder sb = new StringBuilder("(?:");
        for (int i = 0; i < words.length; i++) {
            if (i > 0) {
                sb.append('|');
            }
            sb.append(ci(ascii, words[i]));
        }
        sb.append(')');
        return sb.toString();
    }

    private static boolean isAscii(String s) {
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) > 127) {
                return false;
            }
        }
        return true;
    }

    public FhiDeidentifyTextTuned() {
    }

    private static String subLiteral(Pattern p, String replacement, String input) {
        Matcher m = p.matcher(input);
        StringBuilder sb = new StringBuilder();
        int last = 0;
        while (m.find()) {
            sb.append(input, last, m.start());
            sb.append(replacement);
            last = m.end();
        }
        sb.append(input, last, input.length());
        return sb.toString();
    }

    @Override
    public Object call(Object arg0) {
        if (arg0 == null) {
            return null;
        }
        String text = (String) arg0;
        if (text.length() == 0) {
            return text;
        }
        char nul = (char) 0;
        final Pats p = isAscii(text) ? ASCII : FULL;

        ArrayList found = new ArrayList();
        StringBuilder sb = new StringBuilder();
        Matcher m = p.PLACEHOLDER.matcher(text);
        int last = 0;
        while (m.find()) {
            sb.append(text, last, m.start());
            found.add(m.group());
            sb.append(nul).append("PH").append(found.size() - 1).append(nul);
            last = m.end();
        }
        sb.append(text, last, text.length());
        String working = sb.toString();

        working = subLiteral(p.SSN, "{{SSN}}", working);
        working = subLiteral(p.PHONE, "{{Your Phone Number}}", working);
        working = subLiteral(p.EMAIL, "{{Your Email Address}}", working);
        working = subLiteral(p.DOB, "{{DOB}}", working);
        working = subLiteral(p.MEMBER, "{{MEMBER_ID}}", working);
        working = subLiteral(p.ADDRESS, "{{ADDRESS}}", working);

        if (found.size() > 0) {
            StringBuilder out = new StringBuilder();
            Matcher r = p.RESTORE.matcher(working);
            int prev = 0;
            while (r.find()) {
                out.append(working, prev, r.start());
                String digits = r.group(1);
                // parse Unicode decimal digits, like Python int()
                long idx = 0L;
                boolean huge = false;
                int i = 0;
                while (i < digits.length()) {
                    int cp = digits.codePointAt(i);
                    i += Character.charCount(cp);
                    int dv = Character.digit(cp, 10);
                    if (dv < 0) {
                        dv = 0;
                    }
                    idx = idx * 10L + (long) dv;
                    if (idx > 4000000000L) {
                        huge = true;
                        idx = 4000000000L;
                    }
                }
                if (huge || idx >= (long) found.size()) {
                    out.append(r.group());
                } else {
                    out.append((String) found.get((int) idx));
                }
                prev = r.end();
            }
            out.append(working, prev, working.length());
            working = out.toString();
        }
        return working;
    }
}
