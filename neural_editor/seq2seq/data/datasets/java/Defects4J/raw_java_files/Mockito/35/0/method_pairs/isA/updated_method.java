
    public static <T> T isA(Class<T> clazz) {
        return reportMatcher(new InstanceOf(clazz)).<T>returnFor(clazz);
    }