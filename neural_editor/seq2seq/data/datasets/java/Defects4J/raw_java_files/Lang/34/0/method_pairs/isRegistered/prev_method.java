
    static boolean isRegistered(Object value) {
        Map<Object, Object> m = getRegistry();
        return m.containsKey(value);
    }