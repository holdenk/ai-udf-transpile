package ai_udf;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.sql.api.java.UDF1;

public class FhiParseVerdictLoop implements UDF1<Object, Object> {
    // Python \b...\b around a word-char-delimited alternation, and \s, for str patterns.
    private static final String WS =
        "\\t\\n\\x0B\\f\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000";
    // A lookbehind over a character class steps back one UTF-16 unit, so a
    // supplementary-plane letter or digit before the word reads as its low
    // surrogate (not a word char). {1,2} lets it see the whole code point.
    private static final Pattern WORD = Pattern.compile(
        "(?<![\\p{L}\\p{N}_]{1,2})((?:NOT|UN|NON)[" + WS + "_-]*VIABLE|VIABLE)(?![\\p{L}\\p{N}_])");

    // str.isspace() (all 29 code points are in the BMP)
    private static boolean sp(char c) {
        if (c <= ' ') {
            return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31);
        }
        if (c < 0x85) {
            return false;
        }
        return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A)
            || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    }

    // str.splitlines() boundaries
    private static boolean br(char c) {
        if (c > 0x1E && c < 0x85) {
            return false;
        }
        return c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D || c == 0x1E
            || c == 0x85 || c == 0x2028 || c == 0x2029;
    }

    // re.sub(r"^[\s\-*#>`]*|[\s*`]*$", "", s) on one line: returns {begin, end}
    private static void markup(String s, int[] be) {
        int b = be[0];
        int e = be[1];
        while (b < e) {
            char c = s.charAt(b);
            if (sp(c) || c == '-' || c == '*' || c == '#' || c == '>' || c == '`') {
                b++;
            } else {
                break;
            }
        }
        while (e > b) {
            char c = s.charAt(e - 1);
            if (sp(c) || c == '*' || c == '`') {
                e--;
            } else {
                break;
            }
        }
        be[0] = b;
        be[1] = e;
    }

    @Override
    public Object call(Object arg0) {
        if (arg0 == null) {
            return null;
        }
        String s = (String) arg0;
        int n = s.length();
        int[] be = new int[2];
        int i = 0;
        while (i < n) {
            int b = i;
            while (i < n && !br(s.charAt(i))) {
                i++;
            }
            be[0] = b;
            be[1] = i;
            i++;
            markup(s, be);
            // re.sub(r"^\d+[.)]\s+", "", line): unicode decimal digits, then . or ), then whitespace
            int p = be[0];
            int e = be[1];
            int q = p;
            while (q < e) {
                int cp = s.codePointAt(q);
                if (Character.getType(cp) != Character.DECIMAL_DIGIT_NUMBER) {
                    break;
                }
                q += Character.charCount(cp);
            }
            if (q > p && q < e && (s.charAt(q) == '.' || s.charAt(q) == ')')) {
                int r = q + 1;
                while (r < e && sp(s.charAt(r))) {
                    r++;
                }
                if (r > q + 1) {
                    be[0] = r;
                }
            }
            markup(s, be);
            String upper = s.substring(be[0], be[1]).toUpperCase(Locale.ROOT);
            if (!upper.startsWith("VERDICT")) {
                continue;
            }
            int colon = upper.indexOf(':');
            String value = colon >= 0 ? upper.substring(colon + 1).toUpperCase(Locale.ROOT) : "";
            Matcher m = WORD.matcher(value);
            boolean sawTrue = false;
            boolean sawFalse = false;
            while (m.find()) {
                if (m.group(1).equals("VIABLE")) {
                    sawTrue = true;
                } else {
                    sawFalse = true;
                }
            }
            if (sawTrue == sawFalse) {
                return null;
            }
            return Boolean.valueOf(sawTrue);
        }
        return null;
    }
}
