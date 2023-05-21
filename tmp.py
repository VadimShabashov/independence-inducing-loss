with open('experiment.log', 'r+') as file:
    rows = []
    for row in file:
        if 'Using device' in row or 'Experiment' in row or 'loss =' in row or 'Model with configuration' in row:
            rows.append(row)

    file.seek(0)
    file.writelines(rows)
    file.truncate()

