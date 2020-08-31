
    public static double anyDouble() {
        return reportMatcher(new InstanceOf(Double.class)).returnZero();
    }