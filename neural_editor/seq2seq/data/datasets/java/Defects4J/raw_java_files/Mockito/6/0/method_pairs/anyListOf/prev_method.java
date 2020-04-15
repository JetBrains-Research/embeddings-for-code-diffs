
    public static <T> List<T> anyListOf(Class<T> clazz) {
        return (List) reportMatcher(Any.ANY).returnList();
    }