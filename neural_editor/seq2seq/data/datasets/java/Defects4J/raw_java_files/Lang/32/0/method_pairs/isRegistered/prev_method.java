
    static boolean isRegistered(Object value) {
        return getRegistry().contains(new IDKey(value));
    }