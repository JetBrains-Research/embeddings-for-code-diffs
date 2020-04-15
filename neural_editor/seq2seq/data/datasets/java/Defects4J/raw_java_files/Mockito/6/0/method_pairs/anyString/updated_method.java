
    public static String anyString() {
        return reportMatcher(new InstanceOf(String.class)).returnString();
    }