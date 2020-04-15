
    public static Collection anyCollection() {
        return reportMatcher(new InstanceOf(Collection.class)).returnList();
    }