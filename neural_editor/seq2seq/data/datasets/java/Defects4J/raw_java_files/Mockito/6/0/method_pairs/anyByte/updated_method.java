
    public static byte anyByte() {
        return reportMatcher(new InstanceOf(Byte.class)).returnZero();
    }