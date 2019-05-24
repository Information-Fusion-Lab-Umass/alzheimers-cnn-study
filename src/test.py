from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str,
                                    default="FAILED")
    args = parser.parse_args()
    print("THE RUN ID IS: {}".format(args.run_id))
