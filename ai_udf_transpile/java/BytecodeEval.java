/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Interpreter for the CPython bytecode subset the SQL bytecode backend
 * accepts. One static class, compiled once per UDF jar. No Scala.
 *
 * Opcode numbers are the contract with bytecode_eval.py. Integral math is
 * Java long (silent wrap). % matches Python floor division of the remainder.
 */
package ai_udf;

public final class BytecodeEval {
    public static final int OP_LOAD_FAST = 1;
    public static final int OP_LOAD_CONST = 2;
    public static final int OP_BIN_ADD = 3;
    public static final int OP_BIN_SUB = 4;
    public static final int OP_BIN_MUL = 5;
    public static final int OP_BIN_TRUEDIV = 6;
    public static final int OP_BIN_MOD = 7;
    public static final int OP_UNARY_NEG = 8;
    public static final int OP_UNARY_NOT = 9;
    public static final int OP_CMP_LT = 10;
    public static final int OP_CMP_LE = 11;
    public static final int OP_CMP_EQ = 12;
    public static final int OP_CMP_NE = 13;
    public static final int OP_CMP_GT = 14;
    public static final int OP_CMP_GE = 15;
    public static final int OP_IS = 16;
    public static final int OP_JUMP_IF_FALSE = 17;
    public static final int OP_JUMP_IF_TRUE = 18;
    public static final int OP_JUMP_IF_NONE = 19;
    public static final int OP_JUMP_IF_NOT_NONE = 20;
    public static final int OP_RETURN = 21;
    public static final int OP_POP = 22;
    public static final int OP_COPY = 23;
    public static final int OP_SWAP = 24;

    static final class Frame {
        Object[] stack = new Object[64];
        final Object[] locals = new Object[16];
    }

    static Object[] grow(Frame frame, Object[] stack) {
        frame.stack = java.util.Arrays.copyOf(stack, stack.length * 2);
        return frame.stack;
    }

    static final ThreadLocal<Frame> FRAMES = new ThreadLocal<Frame>() {
        @Override
        protected Frame initialValue() {
            return new Frame();
        }
    };

    private BytecodeEval() {}

    public static Object eval1(int[] code, Object[] consts, Object a) {
        Frame frame = FRAMES.get();
        frame.locals[0] = a;
        try {
            return run(frame, code, consts);
        } finally {
            frame.locals[0] = null;
        }
    }

    public static Object eval2(int[] code, Object[] consts, Object a, Object b) {
        Frame frame = FRAMES.get();
        frame.locals[0] = a;
        frame.locals[1] = b;
        try {
            return run(frame, code, consts);
        } finally {
            frame.locals[0] = null;
            frame.locals[1] = null;
        }
    }

    public static Object evalN(int[] code, Object[] consts, Object[] args) {
        Frame frame = FRAMES.get();
        int n = args.length;
        System.arraycopy(args, 0, frame.locals, 0, n);
        try {
            return run(frame, code, consts);
        } finally {
            for (int i = 0; i < n; i++) {
                frame.locals[i] = null;
            }
        }
    }

    static Object run(Frame frame, int[] code, Object[] consts) {
        Object[] stack = frame.stack;
        Object[] locals = frame.locals;
        int sp = 0;
        int nins = code.length / 2;
        int pc = 0;
        while (pc < nins) {
            int op = code[pc * 2];
            int arg = code[pc * 2 + 1];
            int next = pc + 1;
            switch (op) {
                case OP_LOAD_FAST:
                    if (sp == stack.length) {
                        stack = grow(frame, stack);
                    }
                    stack[sp++] = locals[arg];
                    break;
                case OP_LOAD_CONST:
                    if (sp == stack.length) {
                        stack = grow(frame, stack);
                    }
                    stack[sp++] = consts[arg];
                    break;
                case OP_BIN_ADD: {
                    Object b = stack[--sp];
                    Object a = stack[--sp];
                    stack[sp++] = add(a, b);
                    break;
                }
                case OP_BIN_SUB:
                    sp = binNum(stack, sp, OP_BIN_SUB);
                    break;
                case OP_BIN_MUL:
                    sp = binNum(stack, sp, OP_BIN_MUL);
                    break;
                case OP_BIN_TRUEDIV:
                    sp = binNum(stack, sp, OP_BIN_TRUEDIV);
                    break;
                case OP_BIN_MOD:
                    sp = binNum(stack, sp, OP_BIN_MOD);
                    break;
                case OP_UNARY_NEG: {
                    Object a = stack[--sp];
                    if (a == null) {
                        throw new NullPointerException();
                    }
                    if (integral(a)) {
                        stack[sp++] = Long.valueOf(-asLong(a));
                    } else {
                        stack[sp++] = Double.valueOf(-asDouble(a));
                    }
                    break;
                }
                case OP_UNARY_NOT:
                    stack[sp - 1] = Boolean.valueOf(!truth(stack[sp - 1]));
                    break;
                case OP_CMP_LT:
                case OP_CMP_LE:
                case OP_CMP_EQ:
                case OP_CMP_NE:
                case OP_CMP_GT:
                case OP_CMP_GE: {
                    Object b = stack[--sp];
                    Object a = stack[--sp];
                    stack[sp++] = cmp(a, b, op);
                    break;
                }
                case OP_IS: {
                    Object b = stack[--sp];
                    Object a = stack[--sp];
                    boolean same = a == b;
                    stack[sp++] = Boolean.valueOf(arg == 0 ? same : !same);
                    break;
                }
                case OP_JUMP_IF_FALSE:
                    if (!truth(stack[--sp])) {
                        next = arg;
                    }
                    break;
                case OP_JUMP_IF_TRUE:
                    if (truth(stack[--sp])) {
                        next = arg;
                    }
                    break;
                case OP_JUMP_IF_NONE:
                    if (stack[--sp] == null) {
                        next = arg;
                    }
                    break;
                case OP_JUMP_IF_NOT_NONE:
                    if (stack[--sp] != null) {
                        next = arg;
                    }
                    break;
                case OP_RETURN: {
                    Object result = stack[sp - 1];
                    // Do not retain the last row's objects on a long-lived
                    // executor thread.
                    for (int j = 0; j < sp; j++) {
                        stack[j] = null;
                    }
                    return result;
                }
                case OP_POP:
                    sp--;
                    break;
                case OP_COPY:
                    if (sp == stack.length) {
                        stack = grow(frame, stack);
                    }
                    stack[sp] = stack[sp - arg];
                    sp++;
                    break;
                case OP_SWAP: {
                    int i = sp - arg;
                    Object tmp = stack[sp - 1];
                    stack[sp - 1] = stack[i];
                    stack[i] = tmp;
                    break;
                }
                default:
                    throw new IllegalArgumentException("bad opcode " + op);
            }
            pc = next;
        }
        throw new IllegalStateException("bytecode fell off the end");
    }

    static int binNum(Object[] stack, int sp, int op) {
        Object b = stack[--sp];
        Object a = stack[--sp];
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        boolean both = integral(a) && integral(b);
        long la = 0;
        long lb = 0;
        double da = 0;
        double db = 0;
        if (both && op != OP_BIN_TRUEDIV) {
            la = asLong(a);
            lb = asLong(b);
        } else {
            da = asDouble(a);
            db = asDouble(b);
        }
        switch (op) {
            case OP_BIN_SUB:
                stack[sp++] = both ? Long.valueOf(la - lb) : Double.valueOf(da - db);
                return sp;
            case OP_BIN_MUL:
                stack[sp++] = both ? Long.valueOf(la * lb) : Double.valueOf(da * db);
                return sp;
            case OP_BIN_TRUEDIV:
                if (db == 0.0) {
                    throw new ArithmeticException("division by zero");
                }
                stack[sp++] = Double.valueOf(da / db);
                return sp;
            case OP_BIN_MOD:
                if (both) {
                    if (lb == 0) {
                        throw new ArithmeticException("modulo by zero");
                    }
                    // n % 1 and n % -1 are 0. floorMod(MIN_VALUE, -1) throws.
                    stack[sp++] = (lb == 1 || lb == -1) ? Long.valueOf(0L) : Long.valueOf(Math.floorMod(la, lb));
                } else {
                    if (db == 0.0) {
                        throw new ArithmeticException("modulo by zero");
                    }
                    double r = da % db;
                    if (r == 0.0) {
                        r = Math.copySign(0.0, db);
                    } else if ((da < 0) != (db < 0)) {
                        r += db;
                    }
                    stack[sp++] = Double.valueOf(r);
                }
                return sp;
            default:
                throw new IllegalArgumentException("bad bin " + op);
        }
    }

    static Object add(Object a, Object b) {
        if (a instanceof String || b instanceof String) {
            if (!(a instanceof String) || !(b instanceof String)) {
                throw new IllegalArgumentException("concat");
            }
            return (String) a + (String) b;
        }
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        if (integral(a) && integral(b)) {
            return Long.valueOf(asLong(a) + asLong(b));
        }
        return Double.valueOf(asDouble(a) + asDouble(b));
    }

    static Object cmp(Object a, Object b, int op) {
        if (op == OP_CMP_EQ) {
            return Boolean.valueOf(pyEq(a, b));
        }
        if (op == OP_CMP_NE) {
            return Boolean.valueOf(!pyEq(a, b));
        }
        if (a == null || b == null) {
            throw new NullPointerException();
        }
        int c;
        if (a instanceof String && b instanceof String) {
            c = ((String) a).compareTo((String) b);
        } else if (numeric(a) && numeric(b)) {
            c = cmpNum(a, b);
            if (c == 2) {
                return Boolean.FALSE;
            }
        } else {
            throw new IllegalArgumentException("compare");
        }
        boolean r;
        switch (op) {
            case OP_CMP_LT:
                r = c < 0;
                break;
            case OP_CMP_LE:
                r = c <= 0;
                break;
            case OP_CMP_GT:
                r = c > 0;
                break;
            case OP_CMP_GE:
                r = c >= 0;
                break;
            default:
                throw new IllegalArgumentException("bad cmp " + op);
        }
        return Boolean.valueOf(r);
    }

    static boolean pyEq(Object a, Object b) {
        if (a == null || b == null) {
            return a == b;
        }
        if (a instanceof java.math.BigDecimal && b instanceof java.math.BigDecimal) {
            // equals() is scale-sensitive; 1.0 == 1.00 in python.
            return ((java.math.BigDecimal) a).compareTo((java.math.BigDecimal) b) == 0;
        }
        if (numeric(a) && numeric(b)) {
            return cmpNum(a, b) == 0;
        }
        return a.equals(b);
    }

    // -1, 0, 1, or 2 when unordered (NaN). Long vs double stays exact past 2^53.
    static int cmpNum(Object a, Object b) {
        boolean ai = integral(a);
        boolean bi = integral(b);
        if (ai && bi) {
            return Long.compare(asLong(a), asLong(b));
        }
        double da = asDouble(a);
        double db = asDouble(b);
        if (Double.isNaN(da) || Double.isNaN(db)) {
            return 2;
        }
        if (ai) {
            return cmpLongDouble(asLong(a), db);
        }
        if (bi) {
            return -cmpLongDouble(asLong(b), da);
        }
        if (da == db) {
            return 0;
        }
        return da < db ? -1 : 1;
    }

    static int cmpLongDouble(long n, double d) {
        if (d == Double.POSITIVE_INFINITY) {
            return -1;
        }
        if (d == Double.NEGATIVE_INFINITY) {
            return 1;
        }
        if (d >= 9223372036854775808.0) {
            return -1;
        }
        if (d < -9223372036854775808.0) {
            return 1;
        }
        long trunc = (long) d;
        if (n == trunc && (double) n == d) {
            return 0;
        }
        if (n < trunc) {
            return -1;
        }
        if (n > trunc) {
            return 1;
        }
        return d > 0 ? -1 : 1;
    }

    static boolean truth(Object o) {
        if (o == null || o == Boolean.FALSE) {
            return false;
        }
        if (o instanceof Boolean) {
            return (Boolean) o;
        }
        if (o instanceof Long) {
            return ((Long) o).longValue() != 0L;
        }
        if (o instanceof Integer) {
            return ((Integer) o).intValue() != 0;
        }
        if (o instanceof Double) {
            return ((Double) o).doubleValue() != 0.0;
        }
        if (o instanceof Float) {
            return ((Float) o).floatValue() != 0.0f;
        }
        if (o instanceof Short) {
            return ((Short) o).shortValue() != 0;
        }
        if (o instanceof Byte) {
            return ((Byte) o).byteValue() != 0;
        }
        if (o instanceof java.math.BigDecimal) {
            // decimal(0) must be falsy like python's Decimal('0').
            return ((java.math.BigDecimal) o).signum() != 0;
        }
        if (o instanceof String) {
            return !((String) o).isEmpty();
        }
        return true;
    }

    static boolean numeric(Object o) {
        return o instanceof Long || o instanceof Integer || o instanceof Short || o instanceof Byte
                || o instanceof Double || o instanceof Float || o instanceof Boolean;
    }

    static boolean integral(Object o) {
        return o instanceof Long || o instanceof Integer || o instanceof Short || o instanceof Byte
                || o instanceof Boolean;
    }

    static long asLong(Object o) {
        if (o instanceof Long) {
            return (Long) o;
        }
        if (o instanceof Integer) {
            return (Integer) o;
        }
        if (o instanceof Short) {
            return (Short) o;
        }
        if (o instanceof Byte) {
            return (Byte) o;
        }
        if (o instanceof Boolean) {
            return (Boolean) o ? 1L : 0L;
        }
        throw new IllegalArgumentException("not an int");
    }

    static double asDouble(Object o) {
        if (o instanceof Double) {
            return (Double) o;
        }
        if (o instanceof Float) {
            return (Float) o;
        }
        return asLong(o);
    }
}
