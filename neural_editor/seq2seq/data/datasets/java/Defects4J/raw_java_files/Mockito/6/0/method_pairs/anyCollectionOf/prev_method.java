
    public static <T> Collection<T> anyCollectionOf(Class<T> clazz) {
        return (Collection) reportMatcher(Any.ANY).returnList();
    }