
    public static <T> T anyObject() {
        return (T) reportMatcher(new InstanceOf(Object.class)).returnNull();
    }