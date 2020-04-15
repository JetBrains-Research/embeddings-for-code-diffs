
    public static Map anyMap() {
        return reportMatcher(new InstanceOf(Map.class)).returnMap();
    }