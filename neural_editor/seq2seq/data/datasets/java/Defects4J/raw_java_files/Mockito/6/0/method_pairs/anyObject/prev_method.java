
    public static <T> T anyObject() {
        return (T) reportMatcher(Any.ANY).returnNull();
    }