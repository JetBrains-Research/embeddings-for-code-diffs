
    public static int indexOfAnyBut(String str, String searchChars) {
        if (isEmpty(str) || isEmpty(searchChars)) {
            return INDEX_NOT_FOUND;
        }
        int strLen = str.length();
        for (int i = 0; i < strLen; i++) {
            char ch = str.charAt(i);
            boolean chFound = searchChars.indexOf(ch) >= 0;
            if (i + 1 < strLen && Character.isHighSurrogate(ch)) {
                char ch2 = str.charAt(i + 1);
                if (chFound && searchChars.indexOf(ch2) < 0) {
                    return i;
                }
            } else {
                if (!chFound) {
                    return i;
                }
            }
        }
        return INDEX_NOT_FOUND;
    }