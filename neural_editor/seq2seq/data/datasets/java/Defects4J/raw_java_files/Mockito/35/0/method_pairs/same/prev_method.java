
    public static <T> T same(T value) {
        return reportMatcher(new Same(value)).<T>returnNull();
    }