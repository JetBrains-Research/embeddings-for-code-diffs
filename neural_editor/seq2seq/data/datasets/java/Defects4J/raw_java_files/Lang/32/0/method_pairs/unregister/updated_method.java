
    static void unregister(Object value) {
        Set<IDKey> s = getRegistry();
        if (s != null) {
            s.remove(new IDKey(value));
            synchronized (HashCodeBuilder.class) {
                if (s.isEmpty()) {
                    REGISTRY.remove();
                }
            }
        }
    }