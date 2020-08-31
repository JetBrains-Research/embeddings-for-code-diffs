
    public static int anyInt() {
        return reportMatcher(new InstanceOf(Integer.class)).returnZero();
    }