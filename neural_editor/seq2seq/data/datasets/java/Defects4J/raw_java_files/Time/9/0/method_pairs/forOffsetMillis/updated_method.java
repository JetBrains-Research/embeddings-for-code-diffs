
    public static DateTimeZone forOffsetMillis(int millisOffset) {
        if (millisOffset < -MAX_MILLIS || millisOffset > MAX_MILLIS) {
            throw new IllegalArgumentException("Millis out of range: " + millisOffset);
        }
        String id = printOffset(millisOffset);
        return fixedOffsetZone(id, millisOffset);
    }