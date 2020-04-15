
    public static boolean anyBoolean() {
        return reportMatcher(new InstanceOf(Boolean.class)).returnFalse();
    }