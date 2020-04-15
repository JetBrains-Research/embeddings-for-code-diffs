
    public static <K, V>  Map<K, V> anyMapOf(Class<K> keyClazz, Class<V> valueClazz) {
        return reportMatcher(Any.ANY).returnMap();
    }