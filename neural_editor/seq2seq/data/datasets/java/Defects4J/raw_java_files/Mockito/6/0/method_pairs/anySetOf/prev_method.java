
    public static <T> Set<T> anySetOf(Class<T> clazz) {
        return (Set) reportMatcher(Any.ANY).returnSet();
    }