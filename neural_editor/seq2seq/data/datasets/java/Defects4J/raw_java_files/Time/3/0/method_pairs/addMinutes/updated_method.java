
    public void addMinutes(final int minutes) {
        if (minutes != 0) {
            setMillis(getChronology().minutes().add(getMillis(), minutes));
        }
    }