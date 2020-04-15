
    @SuppressWarnings("deprecation")
    public static LocalDate fromDateFields(Date date) {
        if (date == null) {
            throw new IllegalArgumentException("The date must not be null");
        }
            // handle years in era BC
        return new LocalDate(
            date.getYear() + 1900,
            date.getMonth() + 1,
            date.getDate()
        );
    }