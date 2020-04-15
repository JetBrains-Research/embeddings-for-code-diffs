
    public static char anyChar() {
        return reportMatcher(new InstanceOf(Character.class)).returnChar();
    }