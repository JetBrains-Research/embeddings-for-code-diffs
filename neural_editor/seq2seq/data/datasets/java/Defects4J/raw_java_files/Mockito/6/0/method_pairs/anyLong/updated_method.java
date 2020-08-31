
    public static long anyLong() {
        return reportMatcher(new InstanceOf(Long.class)).returnZero();
    }