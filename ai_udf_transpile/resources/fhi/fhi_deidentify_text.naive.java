package ai_udf;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.sql.api.java.UDF1;

public class FhiDeidentifyText implements UDF1<Object, Object> {

    // Python str-pattern \d : Unicode category Nd
    private static final String D = "\\p{Nd}";
    // Python str-pattern \s (contents of a character class)
    private static final String SC =
        "\\t\\n\\x0B\\f\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000";
    private static final String S = "[" + SC + "]";
    // Python str-pattern \w (contents of a character class)
    private static final String WC = "\\p{L}\\p{N}_";
    // Python \b
    private static final String B =
        "(?:(?<=[" + WC + "])(?![" + WC + "])|(?<![" + WC + "])(?=[" + WC + "]))";
    // Python (?i)[A-Z] / (?i)[a-z]
    private static final String L = "[A-Za-z\\u0130\\u0131\\u017F\\u212A]";

    // Python (?i) literal ASCII letters
    private static String ci(String word) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < word.length(); i++) {
            char c = Character.toLowerCase(word.charAt(i));
            char u = Character.toUpperCase(c);
            sb.append('[').append(c).append(u);
            if (c == 'i') {
                sb.append("\\u0130\\u0131");
            } else if (c == 's') {
                sb.append("\\u017F");
            } else if (c == 'k') {
                sb.append("\\u212A");
            }
            sb.append(']');
        }
        return sb.toString();
    }

    private static String alt(String[] words) {
        StringBuilder sb = new StringBuilder("(?:");
        for (int i = 0; i < words.length; i++) {
            if (i > 0) {
                sb.append('|');
            }
            sb.append(ci(words[i]));
        }
        sb.append(')');
        return sb.toString();
    }

    private static final Pattern PLACEHOLDER = Pattern.compile(
        "\\{\\{(?:FIRST_NAME|LAST_NAME|NAME|PHONE|Your Phone Number|SSN|EMAIL|Your Email Address|DOB|ADDRESS|MEMBER_ID|PLAN_ID|CASEID|SCSID|GPID|FAX|DATE|ID|REDACTED)[^}]*\\}\\}"
        + "|\\[(?:NAME|PHONE|SSN|EMAIL|DOB|ADDRESS|MEMBER_ID|PLAN_ID|FAX|DATE|ID|REDACTED)[^\\]]*\\]");

    private static final Pattern SSN = Pattern.compile(
        B + D + "{3}-" + D + "{2}-" + D + "{4}" + B);

    private static final Pattern PHONE = Pattern.compile(
        "(?<![" + D + "-])(?:\\+?1[-." + SC + "]?)?\\(?" + D + "{3}\\)?[-." + SC + "]?"
        + D + "{3}[-." + SC + "]?" + D + "{4}(?![" + D + "-])");

    private static final Pattern EMAIL = Pattern.compile(
        B + "[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}" + B);

    private static final String MM = "(?:0[1-9]|1[0-2])";
    private static final String DD = "(?:0[1-9]|[12]" + D + "|3[01])";
    private static final String YY = "(?:19|20)" + D + "{2}";

    private static final Pattern DOB = Pattern.compile(
        "(?:" + ci("DOB")
        + "|" + ci("date") + S + "+" + ci("of") + S + "+" + ci("birth")
        + "|" + ci("born")
        + "|" + ci("birthday")
        + "|" + ci("d") + "\\." + ci("o") + "\\." + ci("b") + "\\.?)"
        + S + "*:?" + S + "*"
        + "(?:" + MM + "[/\\-]" + DD + "[/\\-]" + YY
        + "|" + YY + "[/\\-]" + MM + "[/\\-]" + DD + ")");

    private static final Pattern MEMBER = Pattern.compile(
        alt(new String[] {"member", "group", "subscriber", "policy", "plan"})
        + S + "*"
        + "(?:" + ci("id") + "|#|" + ci("number") + "|" + ci("no") + "\\.?)"
        + S + "*:?" + S + "*" + L + "{0,4}" + D + "{6,12}" + B);

    private static final Pattern ADDRESS = Pattern.compile(
        B + D + "{1,5}" + S + "+(?:" + L + L + "+" + S + "*){1,3}"
        + alt(new String[] {"St", "Street", "Ave", "Avenue", "Blvd", "Boulevard", "Dr", "Drive",
                            "Ln", "Lane", "Rd", "Road", "Ct", "Court", "Way", "Pl", "Place",
                            "Cir", "Circle"})
        + "\\.?" + B);

    private static final Pattern RESTORE = Pattern.compile("\\x00PH(" + D + "+)\\x00");

    public FhiDeidentifyText() {
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

        ArrayList found = new ArrayList();
        StringBuilder sb = new StringBuilder();
        Matcher m = PLACEHOLDER.matcher(text);
        int last = 0;
        while (m.find()) {
            sb.append(text, last, m.start());
            found.add(m.group());
            sb.append(nul).append("PH").append(found.size() - 1).append(nul);
            last = m.end();
        }
        sb.append(text, last, text.length());
        String working = sb.toString();

        working = subLiteral(SSN, "{{SSN}}", working);
        working = subLiteral(PHONE, "{{Your Phone Number}}", working);
        working = subLiteral(EMAIL, "{{Your Email Address}}", working);
        working = subLiteral(DOB, "{{DOB}}", working);
        working = subLiteral(MEMBER, "{{MEMBER_ID}}", working);
        working = subLiteral(ADDRESS, "{{ADDRESS}}", working);

        if (found.size() > 0) {
            StringBuilder out = new StringBuilder();
            Matcher r = RESTORE.matcher(working);
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
