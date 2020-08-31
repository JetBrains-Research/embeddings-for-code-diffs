
    public static <T> T same(T value) {
        return (T) reportMatcher(new Same(value)).<T>returnFor((Class) value.getClass());
    }