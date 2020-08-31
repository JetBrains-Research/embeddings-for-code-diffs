
    public static short anyShort() {
        return reportMatcher(new InstanceOf(Short.class)).returnZero();
    }