
    public static boolean containsAny(String cs, String searchChars) {
        if (searchChars == null) {
            return false;
        }
        return containsAny(cs, searchChars.toCharArray());
    }