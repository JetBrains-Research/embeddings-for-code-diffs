
    public static boolean containsAny(CharSequence cs, String searchChars) {
        if (searchChars == null) {
            return false;
        }
        return containsAny(cs, searchChars.toCharArray());
    }