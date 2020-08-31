
    public static <T> T any() {
        return (T) reportMatcher(Any.ANY).returnNull();
    }