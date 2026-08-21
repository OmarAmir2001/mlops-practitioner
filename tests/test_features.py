# tests/test_features.py
def test_customer_schema_produces_correct_feature_count(real_dv):
    from prodml.data_schema import Customer
    example = Customer.model_config["json_schema_extra"]["example"]
    Xt = real_dv.transform([example])
    assert Xt.shape[1] == len(real_dv.get_feature_names_out())