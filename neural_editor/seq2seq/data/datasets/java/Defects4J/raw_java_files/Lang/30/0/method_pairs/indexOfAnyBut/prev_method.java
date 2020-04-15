
    public static int indexOfAnyBut(String str, String searchChars) {
        if (isEmpty(str) || isEmpty(searchChars)) {
            return INDEX_NOT_FOUND;
        }
        int strLen = str.length();
        for (int i = 0; i < strLen; i++) {
            char ch = str.charAt(i);
            if (searchChars.indexOf(ch) < 0) {
                    return i;
            }
        }
        return INDEX_NOT_FOUND;
    }