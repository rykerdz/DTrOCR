import tests.test_arabic_model as ar_model
import tests.test_arabic_processor as ar_processor
import tests.test_model as en_model
import tests.test_processor as en_processor


def run_tests():
    
    print("\nRunning tests on arabic processor!")
    ar_processor.test_tokeniser_preprocess()
    ar_processor.test_image_processor()
    ar_processor.test_tokeniser_with_bos_token()
    ar_processor.test_tokeniser_with_eos_token()
    ar_processor.test_tokeniser_with_eos_and_bos_tokens()

    
    print("Arabic Processor is set!")
    
    print("\nRunning tests on english processor!")
    
    en_processor.test_image_processor()
    en_processor.test_tokeniser_with_bos_token()
    en_processor.test_tokeniser_with_eos_token()
    en_processor.test_tokeniser_with_eos_and_bos_tokens()
    
    print("English Processor is set!")
    
    print("\nRunning tests on arabic model!")
    
    ar_model.test_model()
    ar_model.test_generation_to_be_deterministic()
    ar_model.test_generation_with_and_without_caching()
    
    print("Arabic model is set!")

    print("\nRunning tests on english model!")
    
    en_model.test_model()
    en_model.test_generation_to_be_deterministic()
    en_model.test_generation_with_and_without_caching()

    print("English model is set!")
    

    
    
    
if __name__ == '__main__':
    run_tests()
    