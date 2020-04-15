
    public static <T> T eq(T value) {
        return (T) reportMatcher(new Equals(value)).<T>returnFor((Class) value.getClass());
    }