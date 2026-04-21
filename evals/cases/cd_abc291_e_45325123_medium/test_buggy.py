import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 3\n1 2\n1 3\n4 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 3\n2 1\n3 1\n1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 1\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1\n2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n2 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='237 236\n17 141\n81 191\n114 230\n8 224\n78 180\n196 108\n23 52\n90 131\n66 59\n113 162\n79 18\n40 190\n225 43\n202 54\n212 53\n201 122\n147 129\n33 7\n112 237\n233 179\n216 212\n231 44\n150 117\n215 189\n100 140\n43 166\n5 170\n99 154\n2 106\n193 94\n192 105\n116 161\n200 100\n165 227\n35 26\n139 112\n171 115\n122 145\n110 73\n126 109\n51 206\n109 65\n10 172\n108 49\n42 136\n56 10\n232 142\n59 198\n14 121\n194 114\n36 6\n12 24\n210 27\n52 152\n189 151\n128 229\n7 134\n96 14\n221 96\n45 216\n227 177\n97 158\n71 127\n37 218\n132 232\n169 160\n38 209\n224 118\n68 231\n27 34\n44 64\n29 51\n157 37\n87 82\n205 185\n73 215\n208 58\n105 144\n198 119\n172 182\n195 3\n101 33\n80 214\n206 11\n213 74\n191 130\n160 87\n161 46\n11 124\n103 175\n190 69\n209 95\n18 56\n86 236\n237 70\n211 128\n185 28\n4 133\n22 143\n85 234\n19 25\n140 99\n91 63\n13 165\n16 150\n234 219\n55 195\n34 233\n102 92\n127 156\n69 167\n25 35\n133 107\n115 86\n83 171\n154 47\n106 173\n123 201\n153 168\n76 102\n62 104\n65 66\n89 223\n32 41\n180 97\n82 213\n188 20\n186 207\n64 192\n28 80\n144 31\n236 149\n162 187\n156 76\n167 8\n94 42\n175 1\n21 103\n15 188\n54 68\n214 225\n181 111\n229 199\n235 194\n107 90\n197 22\n149 132\n31 110\n119 16\n142 138\n183 93\n179 159\n219 40\n134 184\n135 83\n20 71\n228 89\n220 196\n158 4\n1 38\n41 23\n138 210\n46 21\n203 91\n163 208\n30 202\n121 81\n223 77\n199 113\n176 67\n117 15\n207 101\n60 125\n39 181\n53 155\n130 211\n57 164\n151 60\n125 226\n218 148\n9 204\n61 176\n187 48\n95 55\n182 36\n152 217\n67 13\n58 221\n47 85\n173 120\n48 5\n3 98\n146 147\n74 139\n155 61\n129 220\n92 9\n77 205\n230 123\n72 39\n226 17\n20 137\n6 183\n63 32\n131 75\n137 186\n26 135\n124 84\n164 71\n148 45\n88 197\n184 88\n136 163\n204 78\n118 228\n49 203\n141 178\n222 200\n178 126\n70 72\n166 146\n104 169\n159 235\n24 57\n50 174\n143 157\n93 12\n111 30\n177 222\n120 153\n174 116\n145 50\n75 2\n98 79\n170 62\n217 19\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
