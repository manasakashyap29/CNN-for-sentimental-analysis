import sys

if __name__ == '__main__':
    answer_fn = sys.argv[1]  # gold labels
    output_fn = sys.argv[2]  # outputs

    answers = []
    with open(answer_fn, 'r') as f:
        for line in f:
            l = line.strip()
            answers.append(int(l))
        f.close()
    
    outputs = []
    with open(output_fn, 'r') as f:
        for line in f:
            l = line.strip()
            outputs.append(int(l))
        f.close()
    
    assert len(answers) == len(outputs), "Output file length mismatches with the answer file length!"
    correct = 0
    total = len(outputs)
    for output, answer in zip(outputs, answers):
        if output == answer:
            correct += 1

    # Print out statistics
    print("Number of test documents = {0:d}".format(total))
    print("Number of correctly-classified test documents = {0:d}".format(correct))
    print("Accuracy = {0:4.1f}%".format(correct / total * 100.))

