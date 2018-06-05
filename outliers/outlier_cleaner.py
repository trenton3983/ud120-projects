def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    error = (net_worths - predictions)**2
    cleaned_data = zip(ages, net_worths, error)
    cleaned_data = sorted(cleaned_data, key=lambda tup: tup[2])

    index = int(len(cleaned_data)*0.9)  # takes the best 90% leaving the worst/top 10%

    cleaned_data = cleaned_data[:index]

    return cleaned_data

