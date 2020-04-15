
    public static long factorial(final int n) {
        long result = Math.round(factorialDouble(n));
        if (result == Long.MAX_VALUE) {
            throw new ArithmeticException(
                    "factorial value is too large to fit in a long");
        }
        return factorials[n];
    }