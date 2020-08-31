
    static void unregister(Object value) {
        getRegistry().remove(new IDKey(value));
    }