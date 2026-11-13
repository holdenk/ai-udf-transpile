package ai_udf;

import org.apache.spark.sql.api.java.UDF1;

public class FhiLegalCheckLoop implements UDF1<Object, Object> {
    // str.isspace() for BMP chars (all 29 are BMP)
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

    @Override
    public Object call(Object arg0) {
        if (arg0 == null) {
            return Boolean.FALSE;
        }
        String s = (String) arg0;
        int n = s.length();
        int i = 0;
        while (i < n) {
            int b = i;
            while (i < n && !br(s.charAt(i))) {
                i++;
            }
            int e = i;
            while (b < e && sp(s.charAt(b))) {
                b++;
            }
            if (e - b >= 6 && s.startsWith("LEGAL:", b)) {
                while (e > b && sp(s.charAt(e - 1))) {
                    e--;
                }
                int vb = b + 6;
                while (vb < e && sp(s.charAt(vb))) {
                    vb++;
                }
                int len = e - vb;
                return Boolean.valueOf(!(len == 0 || (len == 9 && s.startsWith("UNCHECKED", vb))));
            }
            i++;
        }
        return Boolean.FALSE;
    }
}
