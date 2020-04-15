
    public static <T> T eq(T value) {
        return reportMatcher(new Equals(value)).<T>returnNull();
    }