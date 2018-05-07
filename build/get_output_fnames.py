import sys
with open(sys.argv[1], 'rb') as fnf:
    fnames = fnf.readlines()
    out_fnames = []
    for fidx in range(len(fnames)):
        out_fnames.append(sys.argv[2] + '/' + str(fidx)+ '.txt\n')

with open(sys.argv[3], 'wb') as of:
    of.writelines(out_fnames)