# Format for results submission
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': pred_submission})
output.to_csv('submission_2.csv', index=False)