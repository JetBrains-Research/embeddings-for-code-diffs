
    public static List anyList() {
        return reportMatcher(new InstanceOf(List.class)).returnList();
    }