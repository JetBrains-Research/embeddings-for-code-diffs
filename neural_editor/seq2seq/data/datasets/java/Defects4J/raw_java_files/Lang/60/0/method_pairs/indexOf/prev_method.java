
    public int indexOf(char ch, int startIndex) {
        startIndex = (startIndex < 0 ? 0 : startIndex);
        if (startIndex >= size) {
            return -1;
        }
        char[] thisBuf = buffer;
        for (int i = startIndex; i < thisBuf.length; i++) {
            if (thisBuf[i] == ch) {
                return i;
            }
        }
        return -1;
    }