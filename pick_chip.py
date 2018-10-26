# Utility imports
import optparse
import sys
import data

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--cell", action='store', dest='cell_index', type='int',
                      help='Specifies cell with index that corresponds in list file.')
    parser.add_option("--protein", action='store', dest='protein_index', type='int',
                      help='Specifies cell with index that corresponds in list file.')
    (params, _) = parser.parse_args(sys.argv)

    cel_idx = params.cell_index
    pro_idx = params.protein_index

    chip = data.dream_array('chip.conservative.pkl')
    export = chip[:, cel_idx, pro_idx]

    if export[0] == -1:
        print('No data available for cell/protein combo %d/%d' % (cel_idx, pro_idx))
    else:
        data.save_array(export, 'chip-%d-%d.conservative.pkl' % (cel_idx, pro_idx))