
    public static Set anySet() {
        return reportMatcher(new InstanceOf(Set.class)).returnSet();
    }