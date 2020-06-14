#import os; exec(open(os.path.join('D:', 'Source', 'Repos', 'sudoku', 'sudoku.py')).read())
#CD /D D:\Source\Repos\sudoku
#"%ProgramFiles%\Python37\scripts\manim" sudoku.py Sudoku -pl
#"%ProgramFiles%\Python37\scripts\manim" sudoku.py Sudoku -pp
#https://www.sudokuwiki.org/sudoku.htm
#http://hodoku.sourceforge.net/en/techniques.php
#https://staffhome.ecm.uwa.edu.au/~00013890/sudokumin.php
#http://www.afjarvis.staff.shef.ac.uk/sudoku/bertram.html
#https://en.wikipedia.org/wiki/Mathematics_of_Sudoku
#https://theartofmachinery.com/2017/08/14/monte_carlo_counting_sudoku_grids.html
#http://www.afjarvis.staff.shef.ac.uk/sudoku/sud23gp.html
#https://nickp.svbtle.com/sudoku-satsolver
#http://forum.enjoysudoku.com/
import itertools

def orthagonal_pts(i, j): return ((i, j-1), (i-1, j), (i+1, j), (i, j+1))
def filter_bounds_points(l, points):
  return filter(lambda x: x[0] >= 0 and x[0] <= l-1 and x[1] >= 0 and x[1] <= l-1, points)
def orthagonal_points(i, j, l):
  return (frozenset(filter_bounds_points(l, orthagonal_pts(i, j))),)
def diagonal_pts(i, j): return ((i-1, j-1), (i+1, j-1), (i-1, j+1), (i+1, j+1))
def king_rule_pts(i, j): return ((i-1, j-1), (i, j-1), (i+1, j-1), (i-1, j), (i+1, j), (i-1, j+1), (i, j+1), (i+1, j+1))
def king_rule_points(i, j, l):
  return (frozenset(filter_bounds_points(l, king_rule_pts(i, j))),)
"""
def exclude_king_rule(rem, cell_visibility_rules, cell_sudoku_remove):
  l, count = len(rem), 0
  for i in range(l):
    for j in range(l):
      if len(rem[i][j]) != 1:
        points = set(king_rule_points(i, j, l))
        for z in rem[i][j].copy():
          #need to look above, below, left and right to exclude based on column and row rules combined with king rule
          #and combine sub square rule with kings rule
          if (i != 0 and all(not z in rem[i-1][y] or (i-1, y) in points for y in range(l)) or
              i != l-1 and all(not z in rem[i+1][y] or (i+1, y) in points for y in range(l)) or
              j != 0 and all(not z in rem[y][j - 1] or (y, j - 1) in points for y in range(l)) or
              j != l-1 and all(not z in rem[y][j + 1] or (y, j + 1) in points for y in range(l)) or
              i != 0 and sub_square_from_point(i - 1, j) != sub_square_from_point(i, j) and all(not z in rem[y[0]][y[1]] or y[1] == j or y[0] == i - 1 and (y[0], y[1]) in points for y in get_sub_square_points(sub_square_from_point(i - 1, j))) or
              i != l-1 and sub_square_from_point(i + 1, j) != sub_square_from_point(i, j) and all(not z in rem[y[0]][y[1]] or y[1] == j or y[0] == i + 1 and (y[0], y[1]) in points for y in get_sub_square_points(sub_square_from_point(i + 1, j))) or
              j != 0 and sub_square_from_point(i, j - 1) != sub_square_from_point(i, j) and all(not z in rem[y[0]][y[1]] or y[0] == i or y[1] == j - 1 and (y[0], y[1]) in points for y in get_sub_square_points(sub_square_from_point(i, j - 1))) or 
              j != l-1 and sub_square_from_point(i, j + 1) != sub_square_from_point(i, j) and all(not z in rem[y[0]][y[1]] or y[0] == i or y[1] == j + 1 and (y[0], y[1]) in points for y in get_sub_square_points(sub_square_from_point(i, j + 1)))):
            #print("King's Rule Locked Candidates %d in (%d, %d)" % (z, i, j))
            rem, c = cell_sudoku_remove(rem, i, j, z)
            count += 1 + c
            continue
  rem, c = exclude_sudoku_by_group(rem, cell_visibility_rules, cell_sudoku_remove)
  return rem, count + c
"""
def knight_rule_points(i, j, l):
  return (frozenset(filter_bounds_points(l, ((i-2, j-1), (i-2, j+1), (i-1, j-2), (i-1, j+2), (i+1, j-2), (i+1, j+2), (i+2, j-1), (i+2, j+1)))),)
def row_points(i, j, l): return frozenset((i, x) for x in range(l))
def column_points(i, j, l): return frozenset((x, j) for x in range(l))
def subsquare_points(i, j, l): return frozenset(get_sub_square_points(sub_square_from_point(i, j, l), l))
def diagonal_points(i, j, l):
  return frozenset.union(frozenset((x, x) for x in range(l)) if i == j else frozenset(),
                         frozenset((x, l-1-x) for x in range(l)) if i == l - 1 - j else frozenset())
def bishop_rule_pts(i, j, l):
  return frozenset.union(frozenset((i + x, j + x) for x in range(-l+1, l) if x != 0),
                         frozenset((i + x, j - x) for x in range(-l+1, l) if x != 0))
def bishop_rule_points(i, j, l):
  return (frozenset(filter_bounds_points(l, bishop_rule_pts(i, j, l))),)
def row_col_points(i, j, l):
  return (row_points(i, j, l), column_points(i, j, l))
#def standard_sudoku_points(i, j, l):
#  return (row_points(i, j, l), column_points(i, j, l), subsquare_points(i, j, l))
#def standard_sudoku_points(i, j, l):
#  return tuple(c(i, j, l) for c in mutex_regions_to_visibility(standard_sudoku_mutex_regions(l)))
def standard_sudoku_singoverlap_regions(l):
  return (tuple(row_points(x, 0, l) for x in range(l)), tuple(column_points(0, x, l) for x in range(l)))
def standard_sudoku_mutex_regions(l):
  return (*standard_sudoku_singoverlap_regions(l), tuple(get_sub_square_points(x, l) for x in range(l)))
def mutex_regions_to_visibility(regions):
  def points_func_gen(region):
    def points_func(i, j, l):
      for r in region:
        if (i, j) in r: return (r,)
      return ()
    return points_func
  return tuple(points_func_gen(r) for r in regions)
#def standard_sudoku_regions(l):
#  return (*(row_points(x, 0, l) for x in range(l)), *(column_points(0, x, l) for x in range(l)), *(get_sub_square_points(x, l) for x in range(l)))
def diagonal_sudoku_regions(l):
  return tuple((diagonal_points(0, 0, l), diagonal_points(0, l-1, l)))
def cell_visibility(i, j, l, cell_visibility_rules):
  def frozenset_union(val):
    return frozenset() if len(val) == 0 else frozenset.union(*val)
  return frozenset.union(*(frozenset_union(c(i, j, l)) for c in cell_visibility_rules)) - frozenset(((i, j),))
def cell_mutex_visibility(i, j, mutex_rules):
  return (y for x in mutex_rules for y in x if (i, j) in y)
  #return (x - frozenset(((i, j),)) for c in cell_visibility_rules for x in c(i, j, l))
def cell_sudoku_rule(rem, i, j, y, cell_visibility_rules):
  #print("Initialization via Full House/Naked Single %d to (%d, %d)" % (y, i, j))
  l = len(rem)
  for (p, q) in cell_visibility(i, j, l, cell_visibility_rules):
    if y in rem[p][q]: rem[p][q].remove(y)
  for z in rem[i][j].copy():
    if z != y: rem[i][j].remove(z)    
  #for x in range(l):
  #  if x != i and y in rem[x][j]: rem[x][j].remove(y)
  #  if x != j and y in rem[i][x]: rem[i][x].remove(y)
  #for x in get_sub_square_points(sub_square_from_point(i, j)):
  #  if (x[0] != i or x[1] != j) and y in rem[x[0]][x[1]]: rem[x[0]][x[1]].remove(y)
  return rem
def isqrt(n):
  x = n
  y = (x + 1) >> 1
  while y < x:
    x = y
    y = (x + n // x) >> 1
  return x
def sub_square_from_point(i, j, l):
  s = isqrt(l)
  if s * s != l: return -1
  return i // s * s + j // s
def get_sub_square_points(x, l):
  s = isqrt(l)
  if s * s != l: return ()
  i, j = x // s * s, x % s * s
  return frozenset((i + p, j + q) for p in range(s) for q in range(s))
  #return ((i, j), (i, j+1), (i, j+2), (i+1, j), (i+1, j+1), (i+1, j+2), (i+2, j), (i+2, j+1), (i+2, j+2))
def get_rects(width, height, l):
  if l % width != 0 or l % height != 0: return None
  rects = []
  for i in range(0, l, l // width):
    for j in range(0, l, l // height):
      rects.append([(i + k // width, j + k % width) for k in range(l)])
  return rects
def get_jigsaw_rects(rects, l):
  rem = [[0 for _ in range(l)] for _ in range(l)]
  for i, r in enumerate(rects):
    for x in r:
      rem[x[0]][x[1]] = i
  return rem
def check_bad_sudoku(rem):
  if any(any(len(y) == 0 for y in x) for x in rem): return True
  return False
LAST_DIGIT,FULL_HOUSE,NAKED_SINGLE,HIDDEN_SINGLE,LOCKED_CANDIDATES,NAKED_MULTIPLES,HIDDEN_MULTIPLES,BASIC_FISH,FINNED_FISH,X_CHAIN,XY_CHAIN,BIFURCATION=0,1,2,3,4,5,6,7,8,9,10,11
X_CHAIN_SKYSCRAPER,X_CHAIN_TWOSTRINGKITE,X_CHAIN_TURBOT_FISH=0,1,2
KILLER_CAGE_RULE,THERMO_RULE,INEQUALITY_RULE,HIDDEN_CAGE_TUPLE,MIRROR_CAGE_CELL,ORTHAGONAL_NEIGHBOR,MAGIC_SQUARE,SANDWICH_RULE,ARROW_RULE,EVEN_ODD,MIRROR_RULE,SYMMETRY_RULE,BATTLEFIELD_RULE,RENBAN_RULE,IN_ORDER_RULE,JOUSTING_KNIGHTS,FIBONACCI_RULE,DOUBLE_RULE,MANHATTAN_RULE,SMALL_BIG=12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
STRING_RULES = ("Last Digit", "Full House", "Naked Single", "Hidden Single", "Locked Candidates", "Naked %s", "Hidden %s", "Basic %s", "Finned %s", "X-Chain", "XY-Chain", "Bifurcation",
                "Killer Cage Rule", "Thermometer Rule", "Inequality Rule", "Hidden Cage Tuple", "Mirror Cage Cell", "Orthagonal Neighbor", "Magic Square", "Sandwich Rule", "Arrow Rule", "Even or Odd Rule", "Mirror Rule", "Symmetry Rule", "Battlefield Rule", "Renban Rule", "In Order Rule", "Jousting Knights", "Fibonacci Rule", "Double Rule", "Manhattan Rule", "Small or Big Rule")
def naked_single(rem, mutex_rules, cell_visibility_rules, value_set, dynamic_visibility=None):
  l, possible = len(rem), []
  #should find the full houses before naked singles followed by hidden singles
  for i in range(l):
    for j in range(l):
      if len(rem[i][j]) != 1: continue
      y = next(iter(rem[i][j]))
      t = tuple(x for x in cell_visibility(i, j, l, cell_visibility_rules) if y in rem[x[0]][x[1]])
      if not dynamic_visibility is None:
        t = frozenset((*t, *(x for x in dynamic_visibility(i, j, rem, y) if y in rem[x[0]][x[1]])))
      if len(t) != 0: #[(i, j, y) for y in rem[i][j].intersection(vals)]
        classify = NAKED_SINGLE
        for region in cell_mutex_visibility(i, j, mutex_rules):
          tr = tuple(rem[x[0]][x[1]] for x in region if len(rem[x[0]][x[1]]) == 1)
          if len(tr) != 0 and len(set.union(*tr)) == len(value_set) - 1:
            if all(len(rem[x[0]][x[1]]) == 1 for x in frozenset((p, q) for p in range(l) for q in range(l)) - cell_visibility(i, j, l, cell_visibility_rules)) and (
              all(len(rem[x[0]][x[1]]) == 1 or len(rem[x[0]][x[1]]) == 2 and y in rem[x[0]][x[1]] for x in cell_visibility(i, j, l, cell_visibility_rules))):
              classify = LAST_DIGIT
            else:
              classify = FULL_HOUSE
            break
        possible.append(([(x[0], x[1], y) for x in t], classify, ((i,j),))) #or a last digit...
  return possible
def hidden_single(rem, mutex_rules, cell_visibility_rules, value_set, dynamic_mutex_rules=None):
  l, possible = len(rem), []
  regions = tuple(r for regs in mutex_rules for r in regs)
  if not dynamic_mutex_rules is None:
    regions = (*regions, *dynamic_mutex_rules(l))
  for y in value_set:
    for points in regions:
      exclusive = tuple(z for z in points if y in rem[z[0]][z[1]])
      if len(exclusive) == 1:
        if rem[exclusive[0][0]][exclusive[0][1]] == set((y,)): continue
        possible.append(([(exclusive[0][0], exclusive[0][1], z) for z in rem[exclusive[0][0]][exclusive[0][1]] - set((y,))], HIDDEN_SINGLE, (points,)))
  return possible
def locked_candidates(rem, mutex_rules, cell_visibility_rules, value_set, dynamic_visibility=None, dynamic_mutex_rules=None):
  l, possible = len(rem), []
  regions = tuple(r for regs in mutex_rules for r in regs)
  if not dynamic_mutex_rules is None:
    regions = (*regions, *dynamic_mutex_rules(l))
  for points in regions:
    for y in value_set:
      pts = tuple(z for z in points if y in rem[z[0]][z[1]])
      a = tuple(frozenset.union(cell_visibility(z[0], z[1], l, cell_visibility_rules), dynamic_visibility(z[0], z[1], rem, y) if not dynamic_visibility is None else frozenset()) for z in pts)
      s = frozenset.intersection(*a) if len(a) != 0 else frozenset()
      exclude = []
      diff = s.difference(points)
      for o, z in diff:
        if y in rem[o][z]:
          #if points is a row/column then it is claiming, otherwise it is pointing and pointing either on row/column depending on if o or z matches up with any row/column in points
          exclude.append((o, z, y))
      if len(exclude) != 0: possible.append((exclude, LOCKED_CANDIDATES, (points,diff,pts)))
  return possible
def mutex_multiples(rem, mutex_rules, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  for points in (r for regs in mutex_rules for r in regs):
    for p in range(2, len(points)): #naked pairs/triples/quadruples/etc along rows/columns/sub-regions
      for y in itertools.combinations(points, p):
        y_exc = frozenset(points).difference(y)
        s = frozenset.union(*(frozenset(rem[i][j]) for i, j in y))
        if len(s) == p: #s = value_set.difference(s)
          exclude = []
          for i, j in y_exc:
            for q in s.intersection(frozenset(rem[i][j])):
              #print("Naked pairs/triples/quadruples")
              exclude.append((i, j, q))
          if len(exclude) != 0: possible.append((exclude, NAKED_MULTIPLES, (y, s, points)))
        s = s.difference(frozenset.union(*(frozenset(rem[i][j]) for i, j in y_exc)))
        if len(s) == p:
          exclude = []
          for i, j in y:
            for q in frozenset(rem[i][j]).difference(s):
              #print("Hidden pairs/triples/quadruples")
              exclude.append((i, j, q))
          if len(exclude) != 0: possible.append((exclude, HIDDEN_MULTIPLES, (y, s, points)))
  return possible
def basic_fish(rem, mutex_rules, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  regions = tuple(tuple(x) for x in standard_sudoku_singoverlap_regions(l)) #must be completely not visible from each other, but must each be exactly once intersecting another region twice
  for mutex in range(len(regions)):
    m = regions[mutex]
    for y in value_set:
      pre = tuple(frozenset(x for x in n if y in rem[x[0]][x[1]]) for n in m)
      for p in range(2, l >> 1): #complementary fish will always exist for larger ones so only X-Wing, Swordfish and Jellyfish needed, Squirmbag, Whale and Leviathan are not strictly necessary
        for region in itertools.combinations(range(len(m)), p):
          s = tuple(pre[n] for n in region)
          if any(not (1 < len(q) <= p) for q in s): continue
          su = frozenset.union(*s)
          opp = tuple(filter(lambda x: len(su.intersection(x)) != 0, regions[1-mutex]))
          if len(opp) <= p:
            exclude = []
            unn = frozenset.union(*opp).difference(su)
            for r in unn:
              if y in rem[r[0]][r[1]]:
                exclude.append((r[0], r[1], y))
            if len(exclude) != 0: possible.append((exclude, BASIC_FISH, ([m[r] for r in region], s, unn)))
  return possible
  
def finned_fish(rem, mutex_rules, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  regions = tuple(tuple(x) for x in standard_sudoku_singoverlap_regions(l)) #must be completely not visible from each other, but must each be exactly once intersecting another region twice
  for mutex in range(len(regions)):
    m = regions[mutex]
    for y in value_set:
      pre = tuple(frozenset(x for x in n if y in rem[x[0]][x[1]]) for n in m)
      for p in range(2, l >> 1): #complementary fish always will exist like for basic fish for larger ones
        for region in itertools.combinations(range(len(m)), p):
          s = tuple(pre[n] for n in region)
          if any(len(q) <= 1 for q in s): continue #rule out impossible finned fish
          for j in range(p):
            if any(i != j and len(q) > p for i, q in enumerate(s)): continue
            for z in s[j]:
              subs = frozenset(get_sub_square_points(sub_square_from_point(z[0], z[1], l), l))
              if not (0 < len(s[j].difference(subs)) <= p - 1): continue #perhaps this logic needs more detail for really compact subsquare cases?
              su = frozenset.union(*(x.difference(subs) if i == j else x for i, x in enumerate(s)))
              opp = tuple(filter(lambda x: len(su.intersection(x)) != 0, regions[1-mutex]))
              if len(opp) <= p:
                exclude = []
                intsct = subs.intersection(frozenset.union(*opp).difference(su).difference(s[j]))
                for r in intsct:
                  if y in rem[r[0]][r[1]]:
                    exclude.append((r[0], r[1], y))
                if len(exclude) != 0: possible.append((exclude, FINNED_FISH, ([m[r] for r in region], s, intsct)))
  return possible
  
def x_chain(rem, mutex_rules, cell_visibility_rules, value_set, max_depth, dynamic_visibility=None): #solves skyscrapers, empty rectangles (with or without 2 candidates), turbot fish, 2-string kites
  #dual 2-kite strings and dual empty rectangles would not be explicitly identified and are the combination of 2 X-chains
  l, possible = len(rem), []
  def x_chain_rcrse(rem, max_depth, chain, search, exclusions):
    possible = []
    x, y = chain[-1]
    if max_depth == len(chain) - 1: return possible
    for points in cell_mutex_visibility(x, y, mutex_rules):
      s = [p for p in points if search in rem[p[0]][p[1]] and p != (x, y)] #still need to fix by considering all chain values
      if len(s) == 1 and len(exclusions.intersection(s)) == 0: #strong link
        if len(chain) != 1:
          exclude = []
          for p, q in frozenset.intersection(frozenset(cell_visibility(s[0][0], s[0][1], l, cell_visibility_rules)), frozenset(cell_visibility(chain[0][0], chain[0][1], l, cell_visibility_rules))):
            if not (p, q) in chain and (p, q) != s[0] and search in rem[p][q]:
              #4-length X chain can be a skyscraper or empty rectangle
              #print("X-Chain %d in (%d, %d) of length %d" % (search, p, q, len(chain)+1) + " " + str(chain))
              exclude.append((p, q, search))
          if len(exclude) != 0:
            type, fullchain = -1, chain + [s[0]]
            if len(fullchain) == 4:
              ptsx, ptsy = frozenset(x[0] for x in fullchain), frozenset(x[1] for x in fullchain)
              if len(ptsx) == 2 and len(ptsy) == 3 or len(ptsy) == 2 and len(ptsx) == 3:
                type = X_CHAIN_SKYSCRAPER
              elif len(ptsx) == 3 and len(ptsy) == 3 and sub_square_from_point(*fullchain[1], l) == sub_square_from_point(*fullchain[2], l):
                type = X_CHAIN_TWOSTRINGKITE
              else: type = X_CHAIN_TURBOT_FISH
            possible.append((exclude, X_CHAIN, (fullchain,type)))
        cv = cell_visibility(s[0][0], s[0][1], l, cell_visibility_rules)
        if not dynamic_visibility is None:
          cv |= dynamic_visibility(s[0][0], s[0][1], rem, search)
        for point in cv:
          if point in chain or not search in rem[point[0]][point[1]]: continue
          possible.extend(x_chain_rcrse(rem, max_depth, chain + [s[0], point], search, exclusions.union(cv) | frozenset((s[0],))))
    return possible
  for x in range(l):
    for y in range(l):
      if len(rem[x][y]) != 1:
        for z in rem[x][y]:
          possible.extend(x_chain_rcrse(rem, max_depth, [(x, y)], z, frozenset()))
  return possible
def xy_chain(rem, mutex_rules, cell_visibility_rules, value_set, max_depth): #solves XY-Wing/Y-wing, W-Wing and broken triples
  #some bug exists in this code
  l, possible = len(rem), []
  def xy_chain_rcrse(rem, max_depth, chain, search, final):
    possible = []
    x, y = chain[-1]
    if max_depth == len(chain): return possible
    for point in cell_visibility(x, y, l, cell_visibility_rules): #weak link
      nxt = search.intersection(rem[point[0]][point[1]])
      if point in chain or len(rem[point[0]][point[1]]) != 2 or len(nxt) != 1: continue
      #if next(iter(nxt)) == final:
      if next(iter(rem[point[0]][point[1]] - search)) == final:
        exclude = []
        for p, q in frozenset.intersection(frozenset(cell_visibility(point[0], point[1], l, cell_visibility_rules)), frozenset(cell_visibility(chain[0][0], chain[0][1], l, cell_visibility_rules))):
          if (p, q) != chain[0] and (p, q) != chain[-1] and final in rem[p][q]:
            #3-length XY chain is a broken triple or XY-Wing/Y-Wing with pivot as center value
            #4-length XY chain with the outer 2 in the chain and inner 2 in the chain with the same values is a W-Wing
            #print("XY-Chain %d in (%d, %d) of length %d" % (final, p, q, len(chain)+1))
            exclude.append((p, q, final))
        if len(exclude) != 0: possible.append((exclude, XY_CHAIN, (chain + [point], final)))
      possible.extend(xy_chain_rcrse(rem, max_depth, chain + [point], rem[point[0]][point[1]].difference(search), next(iter(search.difference(rem[point[0]][point[1]]))) if final is None else final))
    return possible
  for x in range(l):
    for y in range(l):
      if len(rem[x][y]) == 2:
        possible.extend(xy_chain_rcrse(rem, max_depth, [(x, y)], rem[x][y], None))
  """
  for x in range(l):
    points = get_sub_square_points(x) #because rows/columns have 0 interaction with rows/columns, rows and columns interact at 1 point, subsquares and rows/columns have 3 interactions
    for y in itertools.combinations(range(l), 2): #XY-Wing/Y-Wing/broken triples
      s1, s2 = rem[points[y[0]][0]][points[y[0]][1]], rem[points[y[1]][0]][points[y[1]][1]]
      if len(s1) != 2 or len(s2) != 2 or len(set(s1).intersection(s2)) != 1: continue
      for z in range(l):
        if sub_square_from_point(z, points[y[0]][1]) != sub_square_from_point(x, points[y[0]][1]) and points[y[0]][1] != points[y[1]][1]:
          s3 = rem[z][points[y[0]][1]]
          si = set(s2).intersection(s3)
          if len(s3) == 2 and len(si) == 1 and len(set(s1).intersection(s3)) == 1 and si != set(s1).intersection(s3):
            for q in set(p[0] for p in get_sub_square_points(sub_square_from_point(z, points[y[1]][1]))): #range(z // 3 * 3, z // 3 * 3 + 3):
              if q != points[y[1]][0] and next(iter(si)) in rem[q][points[y[1]][1]]:
                print("Broken Triple %d in (%d, %d) from (%d, %d), (%d, %d), (%d, %d)" % (next(iter(si)), q, points[y[1]][1], points[y[0]][0], points[y[0]][1], points[y[1]][0], points[y[1]][1], z, points[y[0]][1]))
                rem, c = cell_sudoku_remove(rem, q, points[y[1]][1], next(iter(si)))
                count += 1 + c
          s3 = rem[z][points[y[1]][1]]
          si = set(s1).intersection(s3)
          if len(s3) == 2 and len(si) == 1 and len(set(s2).intersection(s3)) == 1 and si != set(s2).intersection(s3):
            for q in set(p[0] for p in get_sub_square_points(sub_square_from_point(z, points[y[0]][1]))): #range(z // 3 * 3, z // 3 * 3 + 3):
              if q != points[y[0]][0] and next(iter(si)) in rem[q][points[y[0]][1]]:
                print("Broken Triple %d in (%d, %d) from (%d, %d), (%d, %d), (%d, %d)" % (next(iter(si)), q, points[y[0]][1], points[y[0]][0], points[y[0]][1], points[y[1]][0], points[y[1]][1], z, points[y[1]][1]))
                rem, c = cell_sudoku_remove(rem, q, points[y[0]][1], next(iter(si)))
                count += 1 + c
        if sub_square_from_point(points[y[0]][0], z) != sub_square_from_point(points[y[0]][0], x) and points[y[0]][0] != points[y[1]][0]:
          s3 = rem[points[y[0]][0]][z]
          si = set(s2).intersection(s3)
          if len(s3) == 2 and len(si) == 1 and len(set(s1).intersection(s3)) == 1 and si != set(s1).intersection(s3):
            for q in set(p[1] for p in get_sub_square_points(sub_square_from_point(points[y[1]][0], z))): #range(z // 3 * 3, z // 3 * 3 + 3):
              if q != points[y[1]][1] and next(iter(si)) in rem[points[y[1]][0]][q]:
                print("Broken Triple %d in (%d, %d) from (%d, %d), (%d, %d), (%d, %d)" % (next(iter(si)), points[y[1]][0], q, points[y[0]][0], points[y[0]][1], points[y[1]][0], points[y[1]][1], points[y[0]][0], z))
                print_candidate_format(rem)
                rem, c = cell_sudoku_remove(rem, points[y[1]][0], q, next(iter(si)))
                count += 1 + c
          s3 = rem[points[y[1]][0]][z]
          si = set(s1).intersection(s3)
          if len(s3) == 2 and len(si) == 1 and len(set(s2).intersection(s3)) == 1 and si != set(s2).intersection(s3):
            for q in set(p[1] for p in get_sub_square_points(sub_square_from_point(points[y[0]][0], z))): #range(z // 3 * 3, z // 3 * 3 + 3):
              if q != points[y[0]][1] and next(iter(si)) in rem[points[y[0]][0]][q]:
                print("Broken Triple %d in (%d, %d) from (%d, %d), (%d, %d), (%d, %d)" % (next(iter(si)), points[y[0]][0], q, points[y[0]][0], points[y[0]][1], points[y[1]][0], points[y[1]][1], points[y[1]][0], z))
                rem, c = cell_sudoku_remove(rem, points[y[0]][0], q, next(iter(si)))
                count += 1 + c
                """
  return possible

"""
def skyscraper(rem, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  for x in range(l):
    for y in value_set: #skyscraper
      matches = set([j for j in range(l) if y in rem[x][j]])
      if len(matches) == 2:
        for z in range(x+1, l):
          subt = set([j for j in range(l) if y in rem[z][j]])
          if len(subt) == 2 and len(matches.intersection(subt)) == 1:
            exclude = []
            for (p, q) in cell_visibility(x, list(matches.difference(subt))[0], l, cell_visibility_rules).intersection(cell_visibility(z, list(subt.difference(matches))[0], l, cell_visibility_rules)):
              if y in rem[p][q]:
                print("Skyscraper %d in (%d, %d)" % (y, p, q))
                exclude.append((p, q, y))
            if len(exclude) != 0: possible.append((exclude, X_CHAIN, (0)))
      matches = set([j for j in range(l) if y in rem[j][x]])
      if len(matches) == 2:
        for z in range(x+1, l):
          subt = set([j for j in range(l) if y in rem[j][z]])
          if len(subt) == 2 and len(matches.intersection(subt)) == 1:
            exclude = []
            for (p, q) in cell_visibility(list(matches.difference(subt))[0], x, l, cell_visibility_rules).intersection(cell_visibility(list(subt.difference(matches))[0], z, l, cell_visibility_rules)):
              if y in rem[p][q]:
                print("Skyscraper %d in (%d, %d)" % (y, p, q))
                exclude.append((p, q, y))
            if len(exclude) != 0: possible.append((exclude, X_CHAIN, (1)))
  return possible
  
def empty_rectangle(rem, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  for x in range(l):
    points = get_sub_square_points(x) #because rows/columns have 0 interaction with rows/columns, rows and columns interact at 1 point, subsquares and rows/columns have 3 interactions
    for y in value_set:
      total = sum(1 for z in points if y in rem[z[0]][z[1]])
      if total == 1: continue
      for p in set(z[0] for z in points):
        for q in set(z[1] for z in points):
          sc, sr = sum(1 for z in points if z[0] == p and y in rem[z[0]][z[1]]), sum(1 for z in points if z[1] == q and y in rem[z[0]][z[1]])
          if sc != 1 and sr != 1 and total == sum(1 for z in points if (z[0] == p or z[1] == q) and y in rem[z[0]][z[1]]):
            excluder, excludec = [], []
            for r in range(l):
              if not (p, r) in points and y in rem[p][r]:
                s = [z for z in range(l) if z != p and not (z, r) in points and y in rem[z][r]]
                if len(s) == 1:
                  if not (s[0], q) in points and y in rem[s[0]][q]:
                    print("Row Empty Rectangle %d in (%d, %d) along (%d, %d)" % (y, s[0], q, p, r))
                    excluder.append((s[0], q, y))
              if not (r, q) in points and y in rem[r][q]:
                s = [z for z in range(l) if z != q and not (r, z) in points and y in rem[r][z]]
                if len(s) == 1:
                  if not (p, s[0]) in points and y in rem[p][s[0]]:
                    print("Column Empty Rectangle %d in (%d, %d) along (%d, %d)" % (y, p, s[0], r, q))
                    excludec.append((p, s[0], y))
            if len(excluder) != 0: possible.append((excluder, X_CHAIN, (0)))
            if len(excludec) != 0: possible.append((excludec, X_CHAIN, (1)))
  return possible
"""

def best_bifurcate(sudoku, mutex_rules, cell_visibility_rules, value_set):
  def bifurcate_inner(sudoku): #to make a proof, must not bifurcate repeatedly without trying all candidates in a cell
    if check_sudoku(sudoku, mutex_rules, value_set)[1]: return False
    if any(any(len(y) == 0 for y in x) for x in sudoku): return True    
    for i in range(l):
      for j in range(l):
        if len(sudoku[i][j]) != 1:
          for z in sudoku[i][j]: #all must not have a solve path
            rem = cell_sudoku_rule([[y.copy() for y in x] for x in sudoku], i, j, z, cell_visibility_rules)
            if any(any(len(y) == 0 for y in x) for x in rem): continue
            rem, _ = sudoku_loop(rem, mutex_rules, cell_visibility_rules, (), value_set)
            if rem is None: continue
            if not bifurcate_inner(rem): return False
          return True
  def bifurcate_depth(sudoku, depth, maxdepth=None):
    if depth == maxdepth: return float("inf")
    if check_sudoku(sudoku, mutex_rules, value_set)[1]: return None
    if any(any(len(y) == 0 for y in x) for x in sudoku): return 0
    mindepth = None
    for i in range(l):
      for j in range(l):
        if len(sudoku[i][j]) != 1:
          depths = []
          for z in sudoku[i][j]:
            rem = cell_sudoku_rule([[y.copy() for y in x] for x in sudoku], i, j, z, cell_visibility_rules)
            if any(any(len(y) == 0 for y in x) for x in rem): depths.append(1); continue
            rem, solve_path = sudoku_loop(rem, mutex_rules, cell_visibility_rules, (), value_set)
            if rem is None: depths.append(1); continue
            res = bifurcate_depth(rem, depth+1, maxdepth if mindepth is None else min(mindepth+1, maxdepth))
            if res is None: return None
            depths.append(1 + res)
          mindepth = max(depths) if mindepth is None else min(mindepth, max(depths))
          #if depth == 0: print(depths)
    return mindepth
  l = len(sudoku)
  depth = [[{} for _ in range(l)] for _ in range(l)]
  print(sum([len(sudoku[i][j]) for i in range(l) for j in range(l) if len(sudoku[i][j]) != 1]))
  for i in range(l):
    for j in range(l):
      if len(sudoku[i][j]) != 1:
        for z in sudoku[i][j]:
          rem = cell_sudoku_rule([[y.copy() for y in x] for x in sudoku], i, j, z, cell_visibility_rules)
          if any(any(len(y) == 0 for y in x) for x in rem): depth[i][j][z] = 1; continue
          rem, solve_path = sudoku_loop(rem, mutex_rules, cell_visibility_rules, (), value_set)
          if rem is None: depth[i][j][z] = 1; print(i, j, z, logical_solve_string(solve_path, mutex_rules)); continue
          if not bifurcate_inner(rem): continue #first need to make sure its a solution e.g. infinite depth
          curdepth = 1
          while True:
            res = bifurcate_depth(rem, 0, curdepth)
            if res != float("inf"):
              if not res is None: depth[i][j][z] = 1 + res
              break
            curdepth += 1
        print(i, j, depth[i][j])
  print(depth)
  return depth
def bifurcate(sudoku, mutex_rules, cell_visibility_rules, value_set): #exclusion so not assuming uniqueness
  best_bifurcate(sudoku, mutex_rules, cell_visibility_rules, value_set)
  def bifurcate_inner(sudoku): #to make a proof, must not bifurcate repeatedly without trying all candidates in a cell
    if check_sudoku(sudoku, mutex_rules, value_set)[1]: return False
    if any(any(len(y) == 0 for y in x) for x in sudoku): return True    
    for i in range(l):
      for j in range(l):
        if len(sudoku[i][j]) != 1:
          for z in sudoku[i][j]: #all must not have a solve path
            rem = cell_sudoku_rule([[y.copy() for y in x] for x in sudoku], i, j, z, cell_visibility_rules)
            if any(any(len(y) == 0 for y in x) for x in rem): continue
            rem, _ = sudoku_loop(rem, mutex_rules, cell_visibility_rules, (), value_set)
            if rem is None: continue
            if not bifurcate_inner(rem): return False
          return True
  l = len(sudoku)
  for i in range(l):
    for j in range(l):
      if len(sudoku[i][j]) != 1:
        for z in sudoku[i][j]:
          rem = cell_sudoku_rule([[y.copy() for y in x] for x in sudoku], i, j, z, cell_visibility_rules)
          if any(any(len(y) == 0 for y in x) for x in rem): return ((((i, j, z),), BIFURCATION, ()),)
          rem, _ = sudoku_loop(rem, mutex_rules, cell_visibility_rules, (), value_set)
          if rem is None: return ((((i, j, z),), BIFURCATION, ()),)
          if bifurcate_inner(rem): return ((((i, j, z),), BIFURCATION, ()),)
  return []
def lookup_mutex(points, mutex_rules):
  for i, regions in enumerate(mutex_rules):
    for j, r in enumerate(regions):
      if points.issubset(r): #if points == r:      
        return i, j
  return None
def get_mutex_string(points, mutex_rules):
  region_names = ("Row", "Column", "Subsquare", "Region")
  x = lookup_mutex(points, mutex_rules)
  if not x is None:
    i, j = x
    subsquare_names = ("Northwest", "Northern", "Northeastern", "Western", "Central", "Eastern", "Southwestern", "Southern", "Southeastern")
    if mutex_rules[i][j] == subsquare_points(next(iter(points))[0], next(iter(points))[1], len(mutex_rules[i][j])):
      return subsquare_names[j] + " " + region_names[i]
    else:
      return region_names[i] + " " + str(j + 1)
  else:
    return "Area" + " " + get_cells_string(points, False)
def get_cell_string(point):
  return "r" + str(point[0] + 1) + "c" + str(point[1] + 1)
def get_cells_string(cells, for_latex):
  if for_latex: return r",\\".join(", ".join(get_cell_string(x) for x in cells[i:i+2]) for i in range(0, len(cells), 2))
  else: return ", ".join(get_cell_string(x) for x in cells)
def get_cell_vals_string(cellvals, for_latex):
  if for_latex: return r",\\".join(", ".join("%d in %s" % (z, get_cell_string((x, y))) for (x, y, z) in cellvals[i:i+2]) for i in range(0, len(cellvals), 2))
  else: return ", ".join("%d in %s" % (z, get_cell_string((x, y))) for (x, y, z) in cellvals)
def logic_step_string(step, mutex_rules, for_latex=False):
  multiples = ("Pairs", "Triples", "Quadruples", "Quintuples", "Sextuples", "Septuples", "Octuples", "Nonuples", "Decuple")
  fish = ("X-Wing", "Swordfish", "Jellyfish", "Squirmbag", "Whale", "Leviathan")
  x_chain_specific = ("Skyscraper", "2-String Kite", "Turbot Fish", "Empty Rectangle")
  name = STRING_RULES[step[1]]
  if step[1] in (LAST_DIGIT, FULL_HOUSE, NAKED_SINGLE):
    logic = "in " + get_cell_string(step[2][0])
  elif step[1] == HIDDEN_SINGLE:
    logic = "along " + get_mutex_string(step[2][0], mutex_rules)
  elif step[1] == LOCKED_CANDIDATES:
    i, _ = lookup_mutex(step[2][0], mutex_rules)
    logic = "in " + get_mutex_string(step[2][0], mutex_rules) + (r"\\" if for_latex else " ") + "cells " + get_cells_string(step[2][2], for_latex) + (r"\\" if for_latex else " ") + ("pointing at" if i != 0 and i != 1 else "claiming from") + (r"\\" if for_latex else " ") + get_mutex_string(step[2][1], mutex_rules)
  elif step[1] == HIDDEN_MULTIPLES or step[1] == NAKED_MULTIPLES:
    name = name % multiples[len(step[2][0])-1-1]
    logic = "in " + get_mutex_string(step[2][2], mutex_rules) + (r"\\" if for_latex else " ") + "cells " + get_cells_string(step[2][0], for_latex) + (r"\\" if for_latex else " ") + "with values" + (r"\\" if for_latex else " ") + ", ".join(str(x) for x in sorted(step[2][1]))
  elif step[1] == BASIC_FISH or step[1] == FINNED_FISH:
    name = name % fish[len(step[2][0])-2]
    logic = "in " + ", ".join(get_mutex_string(x, mutex_rules) for x in step[2][0]) + (r"\\" if for_latex else " ") + "cells " + get_cells_string(tuple(frozenset.union(*step[2][1])), for_latex) + (r"\\" if for_latex else " ") + "for value " + str(step[0][0][2])
  elif step[1] == X_CHAIN:
    if step[2][1] != -1: name = x_chain_specific[step[2][1]]
    logic = "along " + get_cells_string(step[2][0], for_latex) + " with value " + str(step[0][0][2])
  else: logic = str(step[2])
  return name + (r"\\" if for_latex else " ") + logic + (r"\\" if for_latex else " ") + "excludes" + (r"\\" if for_latex else " ") + get_cell_vals_string(step[0], for_latex)
  
def logical_solve_string(steps, mutex_rules):
  return '\n'.join(logic_step_string(x, mutex_rules) for x in steps)

def exclude_sudoku_by_group(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set): #combine rule for row/column with sub regions
  if check_bad_sudoku(rem): return rem, False
  for func in exclude_rule + (naked_single, hidden_single, locked_candidates, mutex_multiples, basic_fish, #skyscraper, empty_rectangle,
               lambda rm, mr, cv, vs: x_chain(rm, mr, cv, vs, 4), finned_fish,
               lambda rm, mr, cv, vs: xy_chain(rm, mr, cv, vs, 3),
               lambda rm, mr, cv, vs: x_chain(rm, mr, cv, vs, 6), lambda rm, mr, cv, vs: xy_chain(rm, mr, cv, vs, None),
               #bifurcate
               ):
    possible = func(rem, mutex_rules, cell_visibility_rules, value_set)
    if len(possible) != 0:
      #print(logic_step_string(possible[0], mutex_rules))
      for (x, y, z) in possible[0][0]:
        #sol = str_to_sudoku("174835269695472138832169475381294756946753821527618394259381647713946582468527913")
        #sol = str_to_sudoku("179524863285316947634978251826143795497852316513769482958231674361497528742685139")
        #sol = str_to_sudoku("178469523463275981592381674357894162216753498849612357634927815921538746785146239")
        #sol = str_to_sudoku("1527346427365157364127364125361257421457636451237")
        #if z == sol[x][y]:
          #print_candidate_format(rem)
          #print(possible[0], STRING_RULES[possible[0][1]])
          #raise ValueError
        if not z in rem[x][y]:
          print(x, y, z, possible)
          raise ValueError
        else: rem[x][y].remove(z)
      #print_logic_step(possible[0])
      return rem, possible[0]
  return rem, None

def sudoku_loop(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set):
  solve_path = []
  while True:
    rem, found = exclude_sudoku_by_group(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set)
    if found is None: break
    solve_path.append(found)
    if any(any(len(y) == 0 for y in x) for x in rem):
      #print(logical_solve_string(solve_path, mutex_rules))
      #print_candidate_format(rem) #print_candidate_border(rem, exc)
      return None, solve_path
  return rem, solve_path

def init_sudoku(sudoku, cell_visibility_rules, value_set):
  l = len(sudoku)
  rem = [[set(value_set) for _ in range(l)] for _ in range(l)]
  for i, x in enumerate(sudoku):
    for j, y in enumerate(x):
      if not y is None:
        rem = cell_sudoku_rule(rem, i, j, y, cell_visibility_rules)
  return rem
  
#len(list(itertools.permutations((1,2,3,4,5,6,7,8,9),9))) == 362880 permutations of numbers 1 to 9
#for 9x9 classic sudoku: 6670903752021072936960 possible grids 5472730538 essentially unique solutions
def solve_sudoku(sudoku, mutex_rules, cell_visibility_rules, exclude_rule, value_set, border_solve):
  l = len(sudoku)
  if not border_solve is None:
    border = border_solve()
    mutex_rules = (*mutex_rules, tuple(frozenset(x) for x in jigsaw_to_coords(border).values()))
  else: border = None
  cell_visibility_rules = mutex_regions_to_visibility(mutex_rules) + cell_visibility_rules
  rem = init_sudoku(sudoku, cell_visibility_rules, value_set)
  rem, solve_path = sudoku_loop(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set)
  #rem = brute_sudoku(rem, cell_visibility_rules, exclude_rule, value_set)
  return rem, solve_path, border
        
#attempt to find where the shallowest brute force is - to discover new logical rule based on that position
#any time a minimum depth is found, one plus that depth automatically covers the rest as it can always go through the minimal route
def brute_sudoku_depth(sudoku, mutex_rules, cell_visibility_rules, exclude_rule, value_set):
  l = len(sudoku)
  def brute_sudoku_depth_rcrse(rem, depth, max_depth=None):
    if check_sudoku(rem, mutex_rules, value_set)[1]: return None
    if any(any(len(y) == 0 for y in x) for x in rem): return depth
    rem, _ = sudoku_loop(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set)
    if rem is None: return depth
    if depth == max_depth: return depth + 1 #at least this much so must indicate...
    d = None
    for i in range(l):
      for j in range(l):
        if len(rem[i][j]) != 1:
          maxDepth = None
          for z in rem[i][j]:
            r = cell_sudoku_rule([[y.copy() for y in x] for x in rem], i, j, z, cell_visibility_rules)
            dp = brute_sudoku_depth_rcrse(r, depth + 1, max_depth if d is None else min(d, max_depth))
            if not dp is None: maxDepth = dp if maxDepth is None else max(maxDepth, dp)
          d = maxDepth if d is None else min(d, maxDepth)
    return d
  depths = [[-1 for _ in range(l)] for _ in range(l)]
  d = None
  finished = None
  for d in range(1, l*l-l-l-1):
    for i in range(l):
      for j in range(l):
        if len(sudoku[i][j]) != 1:
          maxDepth = None
          for z in sudoku[i][j]:
            rem = cell_sudoku_rule([[y.copy() for y in x] for x in sudoku], i, j, z, cell_visibility_rules)
            dp = brute_sudoku_depth_rcrse(rem, 0, d)# if d is None else d + 1)
            #print(i, j, z, dp)
            if not dp is None: maxDepth = dp if maxDepth is None else max(maxDepth, dp)
          d = maxDepth if d is None else min(d, maxDepth)
          depths[i][j] = maxDepth
          if maxDepth == d: finished = maxDepth
          print(i, j, maxDepth, d)
    if not finished is None: break
    print(d)
  return [[(d + 1 if len(sudoku[x][y]) != 1 else 0) if depths[x][y] == -1 else depths[x][y] for y in range(l)] for x in range(l)]

#this generates a possibly solution only as rule does not guarantee uniqueness unless it continues searching
def brute_sudoku(sudoku, mutex_rules, cell_visibility_rules, exclude_rule, value_set):
  if check_sudoku(sudoku, mutex_rules, value_set)[1]: return sudoku
  if any(any(len(y) == 0 for y in x) for x in sudoku): return None
  l = len(sudoku)
  for i in range(l):
    for j in range(l):
      if len(sudoku[i][j]) != 1:
        for z in sudoku[i][j]:
          rem = cell_sudoku_rule([[y.copy() for y in x] for x in sudoku], i, j, z, cell_visibility_rules)
          if any(any(len(y) == 0 for y in x) for x in rem): continue
          rem, _ = sudoku_loop(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set)
          if rem is None: continue
          rem = brute_sudoku(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set)
          if not rem is None: return rem
        return None
#check solution by sum of rows and columns and sub-squares is always 45, each contains a single element
def check_sudoku(rem, mutex_rules, value_set):
  if rem is None: return rem, False
  l = len(rem)
  tot = sum(x for x in value_set)
  if any(any(len(y) != 1 for y in x) for x in rem): return rem, False
  rem = [[next(iter(x)) for x in y] for y in rem]
  for region in mutex_rules:
    for points in region:
      if any(sum(rem[p[0]][p[1]] for p in points) != tot for x in range(l)): return rem, False
  #if any(sum(y for y in x) != tot for x in rem): return rem, False
  #if any(sum(rem[y][x] for y in range(len(rem))) != tot for x in range(l)): return rem, False
  #if any(sum(rem[p[0]][p[1]] for p in get_sub_square_points(x)) != tot for x in range(l)): return rem, False
  return rem, True
def len_reduce(rem): return [[len(x) for x in y] for y in rem]
def print_sudoku(rem):
  l = isqrt(len(rem))
  fmt = " " + (("|" + "%s" * l) * l) + "|\n"
  strs = [fmt % tuple([('.' if len(x) != 1 else str(next(iter(x)))) if isinstance(x, set) else str(x) for x in y]) for y in rem]
  b = "-" * l
  print(" *" + "-".join([b for _ in range(l)]) + "*\n" +
        (" |" + "+".join([b for _ in range(l)]) + "|\n").join(''.join(strs[y * l + x] for x in range(l)) for y in range(l)) +
        " *" + "-".join([b for _ in range(l)]) + "*")
#other format - ":0000:x:..7.3.......56.14.5....1.2...13....229.....643....49...5.7....1.16.59.......1.6..:::"
def print_candidate_format(rem):
  l = isqrt(len(rem))
  r = max(max(len(y) for y in x) for x in rem) + 1
  fmt = " |" + (((" %%-%is" % r) * l) + " |") * l + "\n" #" | %-6s %-6s %-6s | %-6s %-6s %-6s | %-6s %-6s %-6s |\n"
  strs = [fmt % tuple([''.join([str(z) for z in x]) for x in y]) for y in rem]
  b = "-" * (l + 1 + r * l)
  print(" *" + "-".join([b for _ in range(l)]) + "*\n" +
        (" *" + "+".join([b for _ in range(l)]) + "|\n").join(''.join(strs[y * l + x] for x in range(l)) for y in range(l)) +
        " *" + "-".join([b for _ in range(l)]) + "*")

def gen_ominos(l): #fixed polyominos
  base = set((frozenset(((0,0),)),))
  for i in range(l - 1):
    nextbase = set()
    for j in base:
      for (x, y) in j:
        if not (x+1, y) in j: nextbase.add(frozenset((*j, (x+1, y))))
        if not (x-1, y) in j:
          if x-1 < 0: nextbase.add(frozenset((*((p+1,q) for (p, q) in j), (x, y))))
          else: nextbase.add(frozenset((*j, (x-1, y))))
        if not (x, y+1) in j: nextbase.add(frozenset((*j, (x, y+1))))
        if not (x, y-1) in j:
          if y-1 < 0: nextbase.add(frozenset((*((p,q+1) for (p, q) in j), (x, y))))
          else: nextbase.add(frozenset((*j, (x, y-1))))
    base = nextbase
  return base
#[len(gen_ominos(x)) for x in range(1, 10)] == [1, 2, 6, 19, 63, 216, 760, 2725, 9910] #https://oeis.org/A001168
def rotate_omino(o, degree): #degree=0 for 90 degrees, 1 for 180 degrees, 2 for 270 degrees
  mx = max((max(x, y) for (x, y) in o))
  if degree == 0: newpath = frozenset((y, mx-x) for (x, y) in o)
  elif degree == 1: newpath = frozenset((mx-x, mx-y) for (x, y) in o)
  elif degree == 2: newpath = frozenset((mx-y, x) for (x, y) in o)
  mnr = min(newpath, key=lambda x: x[0])[0]
  mnc = min(newpath, key=lambda x: x[1])[1]
  return frozenset((x - mnr, y - mnc) for (x, y) in newpath)
def mirror_omino(o, axis): #axis=0 for x-axis, 1 for y-axis
  mx = max(o, key=lambda x: x[axis])[axis]
  if axis == 0: newpath = frozenset((mx-x, y) for (x, y) in o)
  elif axis == 1: newpath = frozenset((x, mx-y) for (x, y) in o)
  mnr = min(newpath, key=lambda x: x[0])[0]
  mnc = min(newpath, key=lambda x: x[1])[1]
  return frozenset((x - mnr, y - mnc) for (x, y) in newpath)
def one_sided_ominos(l):
  candidates = gen_ominos(l)
  ominos = set()
  for i in candidates:
    if not rotate_omino(i, 0) in ominos and not rotate_omino(i, 1) in ominos and not rotate_omino(i, 2) in ominos:
      ominos.add(i)
  return ominos
#[len(one_sided_ominos(x)) for x in range(1, 10)] == [1, 1, 2, 7, 18, 60, 196, 704, 2500] #https://oeis.org/A000988
def free_ominos(l): #until heptomino with l==7, no holes can occur
  candidates = gen_ominos(l)
  ominos = set()
  for i in candidates:
    if not rotate_omino(i, 0) in ominos and not rotate_omino(i, 1) in ominos and not rotate_omino(i, 2) in ominos:
      m = mirror_omino(i, 0) #can be either axis 0/1 here...
      if not m in ominos and not rotate_omino(m, 0) in ominos and not rotate_omino(m, 1) in ominos and not rotate_omino(m, 2) in ominos:
        ominos.add(i)
  return ominos
#[len(free_ominos(x)) for x in range(1, 10)] == [1, 1, 2, 5, 12, 35, 108, 369, 1285] #https://oeis.org/A000105
def omino_has_hole(o):
  mxr, mxc = max(o, key=lambda x: x[0])[0], max(o, key=lambda x: x[1])[1]
  def get_hole(not_hole, cur_hole, x, y):
    if (x, y) in o or (x, y) in cur_hole: return cur_hole
    if (x, y) in not_hole or x == 0 or x == mxr or y == 0 or y == mxc: return None
    cur_hole.add((x, y))
    if get_hole(not_hole, cur_hole, x+1, y) is None or get_hole(not_hole, cur_hole, x-1, y) is None or get_hole(not_hole, cur_hole, x, y+1) is None or get_hole(not_hole, cur_hole, x, y-1) is None:
      not_hole = not_hole.union(cur_hole)
      return None
    return cur_hole
  not_hole = set()
  for x in range(1, mxr):
    for y in range(1, mxc):
      if not (x, y) in o and not get_hole(not_hole, set(), x, y) is None: return True
      #if not (x, y) in o and (x+1, y) in o and (x-1, y) in o and (x, y+1) in o and (x, y-1) in o: return True
  return False
def free_ominos_hole(l):
  return set(filter(lambda o: omino_has_hole(o), free_ominos(l)))
#[len(free_ominos_hole(x)) for x in range(1, 10)] == [0, 0, 0, 0, 0, 0, 1, 6, 37] #https://oeis.org/A001419
def free_ominos_no_hole(l):
  return set(filter(lambda o: not omino_has_hole(o), free_ominos(l)))
#[len(free_ominos_no_hole(x)) for x in range(1, 10)] == [1, 1, 2, 5, 12, 35, 107, 363, 1248] #https://oeis.org/A001419

def omino_string(o):
  mxr, mxc = max(o, key=lambda x: x[0])[0], max(o, key=lambda x: x[1])[1]
  return ''.join(''.join('#' if (x, y) in o else ' ' for y in range(mxc + 1)) + '\n' for x in range(mxr + 1))
def print_ominos(o):
  for i in o: print(omino_string(i) + '\n')
  
def check_region_point(i, j, points, exc, l):
  if i != 0 and (i-1, j) in points and ((i-1, j), (i, j)) in exc: return False
  if j != 0 and (i, j-1) in points and ((i, j-1), (i, j)) in exc: return False
  if i != l-1 and (i+1, j) in points and ((i, j), (i+1, j)) in exc: return False
  if j != l-1 and (i, j+1) in points and ((i, j), (i, j+1)) in exc: return False
  return True
def check_border(rem, exc):
  l = len(rem)
  for i in range(l):
    for j in range(l):
      if len(rem[i][j]) != 1: continue
      if i != l-1 and rem[i][j] == rem[i+1][j] and ((i, j), (i+1, j)) in exc: return False
      if j != l-1 and rem[i][j] == rem[i][j+1] and ((i, j), (i, j+1)) in exc: return False
  return True

def get_border_count(board, exc):
  l = len(board)
  rem = [[0] * l for _ in range(l)]
  for i in range(l):
    for j in range(l):
      count = 0
      if not board[i][j] is None: continue
      if i == 0 or ((i-1, j), (i, j)) in exc or not board[i-1][j] is None: count += 5 if i == 0 or ((i-1, j), (i, j)) in exc else 1
      if j == 0 or ((i, j-1), (i, j)) in exc or not board[i][j-1] is None: count += 5 if j == 0 or ((i, j-1), (i, j)) in exc else 1
      if i == l-1 or ((i, j), (i+1, j)) in exc or not board[i+1][j] is None: count += 5 if i == l-1 or ((i, j), (i+1, j)) in exc else 1
      if j == l-1 or ((i, j), (i, j+1)) in exc or not board[i][j+1] is None: count += 5 if j == l-1 or ((i, j), (i, j+1)) in exc else 1
      rem[i][j] = count
  return rem
  
def get_hole_path(path, exc, l, cur_hole, x, y):
  if (x, y) in path or (x, y) in cur_hole: return cur_hole
  cur_hole.add((x, y))
  if x != l-1 and not ((x, y), (x+1, y)) in exc: cur_hole = get_hole_path(path, exc, l, cur_hole, x+1, y)
  if x != 0 and not ((x-1, y), (x, y)) in exc: cur_hole = get_hole_path(path, exc, l, cur_hole, x-1, y)
  if y != l-1 and not ((x, y), (x, y+1)) in exc: cur_hole = get_hole_path(path, exc, l, cur_hole, x, y+1)
  if y != 0 and not ((x, y-1), (x, y)) in exc: cur_hole = get_hole_path(path, exc, l, cur_hole, x, y-1)
  return cur_hole
def get_path_boundary(path, l):
  points = set()
  for (x, y) in path:
    if x != l-1 and not (x+1, y) in path: points.add((x+1, y))
    if x != 0 and not (x-1, y) in path: points.add((x-1, y))
    if y != l-1 and not (x, y+1) in path: points.add((x, y+1))
    if y != 0 and not (x, y-1) in path: points.add((x, y-1))
  return points
def region_ominos(point, board, exc):
  l = len(board)
  def filter_holes(path):
    not_hole = set()
    for (x, y) in get_path_boundary(path, l):
      if (x, y) in not_hole: continue
      cur_hole = get_hole_path(path, exc, l, set(), x, y)
      if len(cur_hole) < l: return False
      not_hole = not_hole.union(cur_hole)
    return True
  paths = set((frozenset((point,)),))
  for _ in range(l-1):
    s = set()
    for j in paths:
      for (x, y) in j:
        if x != l-1 and not (x+1, y) in j and not ((x,y), (x+1,y)) in exc and board[x+1][y] is None:
          if ((not (x+1+1, y) in j or not ((x+1, y), (x+1+1, y)) in exc) and
              (not (x+1, y+1) in j or not ((x+1, y), (x+1, y+1)) in exc) and
              (not (x+1, y-1) in j or not ((x+1, y-1), (x+1, y)) in exc)):
            s.add(frozenset((*j, (x+1, y))))
        if x != 0 and not (x-1, y) in j and not ((x-1,y), (x,y)) in exc and board[x-1][y] is None:
          if ((not (x-1-1, y) in j or not ((x-1-1, y), (x-1, y)) in exc) and
              (not (x-1, y+1) in j or not ((x-1, y), (x-1, y+1)) in exc) and
              (not (x-1, y-1) in j or not ((x-1, y-1), (x-1, y)) in exc)):
            s.add(frozenset((*j, (x-1, y))))
        if y != l-1 and not (x, y+1) in j and not ((x,y), (x,y+1)) in exc and board[x][y+1] is None:
          if ((not (x, y+1+1) in j or not ((x, y+1), (x, y+1+1)) in exc) and
              (not (x-1, y+1) in j or not ((x-1, y+1), (x, y+1)) in exc) and
              (not (x+1, y+1) in j or not ((x, y+1), (x+1, y+1)) in exc)):
            s.add(frozenset((*j, (x, y+1))))
        if y != 0 and not (x, y-1) in j and not ((x,y-1), (x,y)) in exc and board[x][y-1] is None:
          if ((not (x, y-1-1) in j or not ((x, y-1-1), (x, y-1)) in exc) and
              (not (x-1, y-1) in j or not ((x-1, y-1), (x, y-1)) in exc) and
              (not (x+1, y-1) in j or not ((x, y-1), (x+1, y-1)) in exc)):
            s.add(frozenset((*j, (x, y-1))))
    paths = s
  return set(filter(filter_holes, paths))
def max_region(paths): #returns guaranteed points, possible points
  if len(paths) == 0: return frozenset(), frozenset()
  return frozenset.intersection(*paths), frozenset.union(*paths)
def add_region(rem, r, num):
  for (i, j) in r:
    rem[i][j] = num
  return rem
def add_region_list(rem, r, num):
  for (i, j) in r:
    if rem[i][j] is None: rem[i][j] = [num]
    else: rem[i][j].append(num)
  return rem
def remove_region_list(rem, r, num):
  for i in range(len(rem)):
    for j in range(len(rem)):
      if not rem[i][j] is None and num in rem[i][j] and not (i, j) in r:
        rem[i][j].remove(num)
        if len(rem[i][j]) == 0: rem[i][j] = None
  return rem
#cannot always use a labeling format since sometimes regions will be clearly inclusive like with 3 walls, but not necesarily distinct or non-distinct from other regions
#start assigning groups to 3 wall spots, then 2 wall spots - corners, then 1 wall spots
#3 walls means that inside the region and the neighbor must be same group
#connectivity - must always have n connected
#can never have same region on two sides of exclusion
#maximum distance - can only go so far in a circular pattern in any direction
def brute_border(exc, l, sudoku, mutex_rules, cell_visibility_rules, exclude_rule, value_set):
  def brute_border_rcrse(path, paths, board):
    if len(paths) == 0:    
      if not check_path_combo(path, set()): return False
      if len(path) == l * (l - 1): return True
      if len(path) == l * (l - 2):
        ominos = region_ominos(next((x, y) for x in range(l) for y in range(l) if board[x][y] is None), board, exc)
        return any(brute_border_rcrse(path, [[o]], board) for o in ominos)
      else: return True
    ret = False
    for x in paths[0]:
      u = path.union(x)
      if len(u) != len(path) + len(x): continue
      nextboard = add_region([y[:] for y in board], x, len(u) // l)
      if brute_border_rcrse(u, paths[1:], nextboard):
        if len(u) == l * (l - 1):
          finalpaths = [set() for _ in range(l)]
          for i in range(l):
            for j in range(l):
              finalpaths[(l if nextboard[i][j] is None else nextboard[i][j]) - 1].add((i, j))
          rem = solve_sudoku(sudoku, (*mutex_rules, tuple(frozenset(x) for x in finalpaths)), cell_visibility_rules, exclude_rule, value_set, None)[0]
          if not rem is None and check_sudoku(rem, mutex_rules, value_set)[1]:
            ret = True
        else: ret = True
    return ret
  def check_path_valid(path, board, val):
    return all(board[x][y] is None or board[x][y] == val for (x, y) in path)
  def check_path_combo(path, combo):
    u = path.union(combo)
    if len(u) != len(path) + len(combo): return False
    not_hole = set()
    for (x, y) in get_path_boundary(u, l):
      if (x, y) in not_hole: continue
      cur_hole = get_hole_path(u, exc, l, set(), x, y)
      if len(cur_hole) % l != 0: return False
      if len(cur_hole) == l: #if last l, must not have any exclusions within it
        for e in itertools.combinations(sorted(cur_hole), 2):
          if e in exc: return False
      #else: #are there not any path combinations in the hole to force exclusions
        #ominos = region_ominos(next(iter(cur_hole)), add_region([[None] * l for _ in range(l)], u, 1), exc)
        #if not any(check_path_combo(u, o) for o in ominos): return False
      not_hole = not_hole.union(cur_hole)
    return True

  board = [[None] * l for _ in range(l)]
  certain_board = [[None] * l for _ in range(l)]
  ominos = []
  while True:
    cur_borders = get_border_count(board, exc)
    mx = max((max(x) for x in cur_borders))
    if mx == 0: break
    for i in range(l):
      for j in range(l):
        if cur_borders[i][j] == mx:
          ominos.append(region_ominos((i, j), certain_board, exc))
          maxes = max_region(ominos[-1])
          certain_board = add_region(certain_board, maxes[0], len(ominos))
          board = add_region_list(board, maxes[1], len(ominos))
          #if len(maxes[0]) != 1:
          for x in range(len(ominos)):
            ominos[x] = set(filter(lambda o: check_path_valid(o, certain_board, x+1), ominos[x]))
            ominos[x] = set(filter(lambda o: all(any(check_path_combo(o, z) for z in y) for idx, y in enumerate(ominos) if x != idx), ominos[x]))
            ominos[x] = set(filter(lambda o: brute_border_rcrse(o, ominos[:x] + ominos[x+1:], add_region([[None] * l for _ in range(l)], o, 1)), ominos[x]))
            maxes = max_region(ominos[x])
            certain_board = add_region(certain_board, maxes[0], x+1)
            board = remove_region_list(board, maxes[1], x+1)
          break
      else: continue
      break
    else: break
  last_omino = tuple((i, j) for i in range(l) for j in range(l) if certain_board[i][j] is None)
  if len(last_omino) == l:
    ominos.append({frozenset(last_omino)})
    maxes = max_region(ominos[-1])
    add_region(certain_board, maxes[0], l)
    board = add_region_list(board, maxes[0], l)
  return board, certain_board, ominos

def print_border(rem, exc):
  l, s = len(rem), ""
  for i in range(l):
    s += (" |" +
      "".join((("." if len(rem[i][j]) != 1 else str(next(iter(rem[i][j])))) if isinstance(rem[i][j], set) else (' ' if rem[i][j] is None else str(rem[i][j]))) + ("|" if ((i, j), (i, j+1)) in exc else " ") for j in range(l-1)) +
      (("." if len(rem[i][l-1]) != 1 else str(next(iter(rem[i][l-1])))) if isinstance(rem[i][l-1], set) else (' ' if rem[i][l-1] is None else str(rem[i][l-1]))) + "|\n")
    if i != l-1: s += (" " + ("+" if ((i, 0), (i+1, 0)) in exc else "|") +
      "".join(("" if j == 0 else (
                ("+" if ((i, j-1), (i, j)) in exc or ((i+1, j-1), (i+1, j)) in exc else "-")
                if ((i, j), (i+1, j)) in exc else
                (("+" if ((i, j-1), (i+1, j-1)) in exc else "|") if ((i, j-1), (i, j)) in exc or ((i+1, j-1), (i+1, j)) in exc else ("-" if ((i, j-1), (i+1, j-1)) in exc else " ")))) +
              ("-" if ((i, j), (i+1, j)) in exc else " ") for j in range(l)) +
      ("+" if ((i, l-1), (i+1, l-1)) in exc else "|") + "\n")
  print(" *-" + "-".join("+" if ((0, j), (0, j+1)) in exc else "-" for j in range(l-1)) + "-*\n" + s +
    " *-" + "-".join("+" if ((l-1, j), (l-1, j+1)) in exc else "-" for j in range(l-1)) + "-*")

def exc_from_border(rem, exc):
  exc = set(exc)
  for i in range(len(rem)):
    for j in range(len(rem)):
      if i != len(rem) - 1 and not rem[i][j] is None and not rem[i+1][j] is None and rem[i][j] != rem[i+1][j] and not ((i, j), (i+1, j)) in exc:
        exc.add(((i, j), (i+1, j)))
      if j != len(rem) - 1 and not rem[i][j] is None and not rem[i][j+1] is None and rem[i][j] != rem[i][j+1] and not ((i, j), (i, j+1)) in exc:
        exc.add(((i, j), (i, j+1)))
  return exc
def print_candidate_border(rem, exc):
  l, s = len(rem), ""
  r = max(max(len(y) for y in x) for x in rem) + 1
  b = "-" * (r + 1)
  fmt = " %%-%is" % r
  for i in range(l):
    s += (" |" +
      "".join((" " * (r + 1) if rem[i][j] is None else fmt % ''.join(str(x) for x in rem[i][j])) + ("|" if ((i, j), (i, j+1)) in exc else " ") for j in range(l-1)) +
      (" " * (r + 1) if rem[i][l-1] is None else fmt % ''.join(str(x) for x in rem[i][l-1])) + "|\n")
    if i != l-1: s += (" " + ("+" if ((i, 0), (i+1, 0)) in exc else "|") +
      "".join(("" if j == 0 else (
                ("+" if ((i, j-1), (i, j)) in exc or ((i+1, j-1), (i+1, j)) in exc else "-")
                if ((i, j), (i+1, j)) in exc else
                (("+" if ((i, j-1), (i+1, j-1)) in exc else "|") if ((i, j-1), (i, j)) in exc or ((i+1, j-1), (i+1, j)) in exc else ("-" if ((i, j-1), (i+1, j-1)) in exc else " ")))) +
              (b if ((i, j), (i+1, j)) in exc else " " * (r + 1)) for j in range(l)) +
      ("+" if ((i, l-1), (i+1, l-1)) in exc else "|") + "\n")
  print(" *" + b + b.join("+" if ((0, j), (0, j+1)) in exc else "-" for j in range(l-1)) + b + "*\n" + s +
    " *" + b + b.join("+" if ((l-1, j), (l-1, j+1)) in exc else "-" for j in range(l-1)) + b + "*")

def jigsaw_to_coords(jigsaw):
  coords = dict()
  for i, x in enumerate(jigsaw):
    for j, y in enumerate(x):
      if not y in coords: coords[y] = set()
      coords[y].add((i, j))
  return coords
def killer_to_jigsaw(killer, l):
  return cages_to_jigsaw([y for _, y in killer], l)
def cages_to_jigsaw(cages, l):
  rem = [[0 for _ in range(l)] for _ in range(l)]
  for i, x in enumerate(cages):
    for z in x:
      rem[z[0]][z[1]] = i + 1
  return rem

#print_border(killer_to_jigsaw(killer_xxl_sudoku, 9), exc_from_border(killer_to_jigsaw(killer_xxl_sudoku, 9), frozenset()))
#print_border(((None,) * 9,) * 9, exc_from_border(killer_to_jigsaw(killer_xxl_sudoku, 9), frozenset()))

def exclude_magic_square_rule_gen(magic_squares):
  def center(ms, s):
    return ms[s >> 1][s >> 1]
  def diagonals(ms, s):
    return frozenset([ms[i][i] for i in range(s)] + [ms[i][s-1-i] for i in range(s)]) - (frozenset() if s & 1 == 0 else frozenset((center(ms, s),)))
  def orthagonals(ms, s):
    l = set()
    for i in range(s):
      for j in range(s):
        if i == j or i == s-1-j: continue
        l.add(ms[i][j])
    return l
    #return [i for i in ms] + [tuple(ms[j][i] for j in range(s)) for i in range(s)]
  def exclude_magic_square_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    s = isqrt(len(value_set))
    if s * s != l: return possible
    combs = get_all_mutex_digit_sum(sum(value_set) // s, s, value_set)
    #[(1, 5, 9), (1, 6, 8), (2, 4, 9), (2, 5, 8), (2, 6, 7), (3, 4, 8), (3, 5, 7), (4, 5, 6)]
    vals = {k: len(tuple(c for c in combs if k in c)) for k in value_set}
    r = {k for k in vals if vals[k] >= 4}
    n = {k for k in vals if vals[k] == 3}
    q = {k for k in vals if vals[k] == 2}
    #middle is part of 4 sums, 4 corners are part of 3 sums, remaining 4 part of 2 sums
    #odd magic squares: center has 4 shared values
    #all magic squares: all diagonals have 3 shared values, all others 2 shared values
    if len(r) == 1 and len(n) == 4 and len(q) == 4:
      for ms in magic_squares:
        exclude = []
        if s & 1 == 1:
          c = center(ms, s)
          for y in rem[c[0]][c[1]].difference(r):
            exclude.append((c[0], c[1], y))
          for p in diagonals(ms, s):
            for y in rem[p[0]][p[1]].difference(n):
              exclude.append((p[0], p[1], y))
          for p in orthagonals(ms, s):
            for y in rem[p[0]][p[1]].difference(q):
              exclude.append((p[0], p[1], y))
        if len(exclude) != 0: possible.append((exclude, MAGIC_SQUARE, ()))
    return possible
  return exclude_magic_square_rule
def exclude_inequality_rule_gen(inequalities): #this is just a special case of 2-cell thermometer
  def exclude_inequality_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    exclude = []
    for iq in inequalities:
      y = max(rem[iq[0][0]][iq[0][1]])
      for x in rem[iq[1][0]][iq[1][1]]:
        if x >= y: exclude.append((iq[1][0], iq[1][1], x))
      y = min(rem[iq[1][0]][iq[1][1]])
      for x in rem[iq[0][0]][iq[0][1]]:
        if x <= y: exclude.append((iq[0][0], iq[0][1], x))
    if len(exclude) != 0: possible.append((exclude, INEQUALITY_RULE, ()))
    return possible
  return exclude_inequality_rule
def exclude_thermo_rule_gen(thermo):
  def exclude_thermo_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    #an initial placement function could easily use puzzle length minus thermometer lengths to place all possible values much more simply as an optimization
    l, possible = len(rem), []
    exclude = []
    for t in thermo:
      lt = len(t)
      for i in range(lt):
        for y in rem[t[i][0]][t[i][1]]:
          lasty = y
          for j in range(i+1, lt): #ascending check
            check = {x for x in rem[t[j][0]][t[j][1]] if x > lasty}
            if len(check) == 0:
              exclude.append((t[i][0], t[i][1], y))
              break
            lasty = min(check)
          else:
            lasty = y
            for j in range(i-1, -1, -1): #descending check
              check = {x for x in rem[t[j][0]][t[j][1]] if x < lasty}
              if len(check) == 0:
                exclude.append((t[i][0], t[i][1], y))
                break
              lasty = max(check)
    if len(exclude) != 0: possible.append((exclude, THERMO_RULE, ()))
    return possible
  return exclude_thermo_rule
def exclude_arrow_rule_gen(arrows, increasing):
  def exclude_arrow_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    maxarrow = max(value_set)
    mins = [] #mins nearest rounded down, maxes nearest rounded up
    for x in sorted(value_set):
      if len(mins) == 0: mins.append(x)
      else: mins.append(mins[-1] + x)
    for a in arrows:
      #if len(a) == 1: #cannot happen as it would be have to be 0 or a broken arrow
      #if len(a) == 2: #equal values, only possible on diagonal
      minarrow = mins[len(a) - 1 - 1]
      possibles = [set() for _ in range(len(a))]
      for x in set(range(minarrow, maxarrow + 1)).intersection(rem[a[0][0]][a[0][1]]):
        sols = get_all_mutex_digit_sum_sets(x, [rem[p[0]][p[1]] for p in a[1:]], set((x,)))
        if increasing: sols = list(filter(lambda s: all(s[i-1] > s[i] for i in range(1,len(s))), sols)) #increases from point of arrow to circle 
        if len(sols) != 0:
          possibles[0].add(x)
          for s in sols:
            for i, y in enumerate(s):
              possibles[1+i].add(y)
      exclude = []
      for i, p in enumerate(a):
        for y in rem[p[0]][p[1]].difference(possibles[i]):
          exclude.append((p[0], p[1], y))
      if len(exclude) != 0: possible.append((exclude, ARROW_RULE, ()))
    return possible
  return exclude_arrow_rule
def add_multipoly(mp1, mp2):
  mp1keys, mp2keys, res = mp1.keys(), mp2.keys(), dict()
  for p in mp1keys & mp2keys: res[p] = mp1[p] + mp2[p]
  for p in mp1.keys() - mp2.keys(): res[p] = mp1[p]
  for p in mp2.keys() - mp1.keys(): res[p] = mp2[p]
  return res
def mul_multipoly(mp1, mp2, l):
  res = dict()
  for p1 in mp1:
    for p2 in mp2:
      idx = tuple(p1[i]+p2[i] for i in range(l))
      if idx in res: res[idx] += mp1[p1] * mp2[p2]
      else: res[idx] = mp1[p1] * mp2[p2]
  return res
def num_combs(total, digits, value_set):
  #the number of ways to make a sum of m in a cage of n cells is the coefficient of t^n x^m in the expansion of the polynomial (1 + tx) (1 + tx^2) ... (1 + tx^9)
  multipoly = {(0, 0):1} #(n, m):(a, b) of a*t^n*x^m
  for v in sorted(value_set):
    #multiply by 1+tx^v or ((1, v):1) then sum them - multiply by one is trivial
    multipoly = add_multipoly(multipoly, mul_multipoly(multipoly, {(1, v):1}, 2))
  return multipoly[(digits, total)] if (digits, total) in multipoly else None
def num_multi_cage_combs(total, digit_vals, value_set): #does not handle dependencies
  #the number of ways to make a sum of m in cages of n and p cells is the coefficient of t^n u^p x^m in the expansion of the polynomial (1 + tx) (1 + tx^2) ... (1 + tx^9)(1 + ux) (1 + ux^2) ... (1 + ux^9)
  l = len(digit_vals)
  finpoly = None
  multipoly = {(0, 0):1} #(n, m):(a, b) of a*t^n*x^m
  for v in sorted(value_set):
    #multiply by 1+tx^v or ((1, v):1) then sum them - multiply by one is trivial
    multipoly = add_multipoly(multipoly, mul_multipoly(multipoly, {(1, v):1}, 2))
  for i, digits in enumerate(digit_vals):
    n = {(*(k[0] if i == j else 0 for j in range(l)), k[-1]):val for k, val in multipoly.items() if k[0] == digits}
    finpoly = n if finpoly is None else mul_multipoly(finpoly, n, l + 1)
  return finpoly[(*digit_vals, total)] if (*digit_vals, total) in finpoly else None
def num_multi_cage_combs_sets(total, digit_vals, value_sets): #does not handle dependencies
  #the number of ways to make a sum of m in cages of n and p cells is the coefficient of t^n u^p x^m in the expansion of the polynomial (1 + tx) (1 + tx^2) ... (1 + tx^9)(1 + ux) (1 + ux^2) ... (1 + ux^9)
  l = len(digit_vals)
  finpoly = None
  for i, digits in enumerate(digit_vals):
    multipoly = {(0, 0):1} #(n, m):(a, b) of a*t^n*x^m
    for v in sorted(value_sets[i]):
      #multiply by 1+tx^v or ((1, v):1) then sum them - multiply by one is trivial
      multipoly = add_multipoly(multipoly, mul_multipoly(multipoly, {(1, v):1}, 2))
    n = {(*(k[0] if i == j else 0 for j in range(l)), k[-1]):val for k, val in multipoly.items() if k[0] == digits}
    finpoly = n if finpoly is None else mul_multipoly(finpoly, n, l + 1)
  return finpoly[(*digit_vals, total)] if (*digit_vals, total) in finpoly else None
def multi_cage_sums(total, digit_vals, value_set):
  #in this case with a single value set the min and max is trivial and does not require a complicated routine...
  mins, maxes = [min_sum_efficient([set(value_set) for _ in d]) for d in digit_vals if len(d) != 0], [max_sum_efficient([set(value_set) for _ in d]) for d in digit_vals if len(d) != 0]
  minmaxes = [None if (mn, mx) == (None, None) else set(range(mn[0], mx[0]+1)) for mn, mx in zip(mins, maxes)]  
  def multi_cage_sums_inner(totals, digit_vals, value_sets):
    if len(totals) == 0: return []
    combs = get_all_mutex_digit_sum(totals[0], len(digit_vals[0]), value_sets[0])
    groupdep = {}
    for i, d in enumerate(digit_vals[0]):
      for x in d:
        if not x-1 in groupdep: groupdep[x-1] = [i]
        else: groupdep[x-1].append(i)
    perms = []
    for c in combs:
      for p in itertools.permutations(c):
        inner = multi_cage_sums_inner(totals[1:], digit_vals[1:], [vs.difference({p[x] for x in groupdep[i]}) if i in groupdep else vs for i, vs in enumerate(value_sets[1:])])        
        if not inner is None:
          if len(inner) == 0: perms.append(p)
          else: perms.extend([(*p, *i) for i in inner])
    return perms if len(perms) != 0 else None
  combs = []
  for totals in get_all_mutex_digit_group_sum(total, [1 for _ in minmaxes], minmaxes):
    inner = multi_cage_sums_inner(totals, [d for d in digit_vals if len(d) != 0], [value_set for d in digit_vals if len(d) != 0])
    if not inner is None: combs.extend(inner)
  sumsets = [set() for _ in digit_vals]
  for c in combs:
    idx = 0
    sumset = [0 for _ in digit_vals]
    for i, d in enumerate(digit_vals):
      for x in d:
        sumset[i] += c[idx]
        for dep in x:
          sumset[i+dep] += c[idx]
        idx += 1
    for i, s in enumerate(sumset): sumsets[i].add(s)
  return sumsets
def get_all_mutex_digit_sum(total, digits, value_set):
  combs = []
  for comb in itertools.combinations(value_set, digits):
    if sum(comb) == total: combs.append(comb)
  return combs
def get_all_mutex_digit_group_sum(total, digits, value_sets):
  combs = []
  if len(digits) == 0: return combs if total == 0 else None
  if total <= 0: return None
  for comb in itertools.combinations(value_sets[0], digits[0]):
    innercombs = get_all_mutex_digit_group_sum(total - sum(comb), digits[1:], value_sets[1:])
    if not innercombs is None: combs += [(*comb, *x) for x in innercombs] if len(innercombs) != 0 else [comb]
  return combs if len(combs) != 0 else None
def get_all_mutex_digit_sum_rcrse(total, digits, values): #more efficient due to exclusions
  def get_all_mutex_digit_sum_inner(total, digits, values):
    if digits == 1: return [(total,)] if total in values else []
    if total <= 0: return []
    l, srt = [], list(sorted(values))
    minrem = sum(srt[:digits-1])
    for y in reversed(srt):
      if y + minrem <= total:
        l.extend([(*x, y) for x in get_all_mutex_digit_sum_inner(total - y, digits - 1, {v for v in values if v < y})])# values - set((y,)))])
    return l
  return get_all_mutex_digit_sum_inner(total, digits, values)
def get_all_mutex_digit_sum_sets(total, value_sets, used_values):
  #get_all_mutex_digit_sum_sets(43, [[x for x in range(2, 10)] for _ in range(7)], set((1,)))
  def get_all_mutex_digit_sum_inner(total, value_sets, used_values):
    if len(value_sets) == 1: return [(total,)] if total in value_sets[0] and not total in used_values else []
    if total <= 0: return []
    l = []
    for y in value_sets[0]:
      if not y in used_values:
        l.extend([(y, *x) for x in get_all_mutex_digit_sum_inner(total - y, value_sets[1:], used_values | set((y,)))])
    return l
  return get_all_mutex_digit_sum_inner(total, value_sets, used_values)
def find_dep_min_sum():
  import secrets
  import random
  while True:
    i = secrets.choice((2,3,4,5,6,7,8,9))
    z = [set(random.sample({1,2,3,4,5,6,7,8,9}, secrets.choice((2,3,4,5,6,7,8,9)))) for _ in range(i)]
    mn = min_sum(z, set())
    mneff = min_sum_efficient(z)
    if mn != mneff: print(mn, mneff, z)

def find_sum_efficient(total, value_sets):
  combs = []
  for comb in itertools.combinations(sorted(set.union(*value_sets)), len(value_sets)):
    if sum(comb) != total: continue
    v = list(vs.intersection(comb) for vs in value_sets)
    #now if any 1 has 0 values continue
    if any(len(vs) == 0 for vs in v): continue
    for p in range(2, len(value_sets)):
      for c in itertools.combinations(v, p):
        if len(set.union(*c)) < p: #any p of the sets together have less than p values
          break
      else: continue
      break
    else: combs.append(comb)
  return combs
def min_sum_efficient(value_sets):
  for comb in itertools.combinations(sorted(set.union(*value_sets)), len(value_sets)):
    v = list(vs.intersection(comb) for vs in value_sets)
    #now if any 1 has 0 values continue
    if any(len(vs) == 0 for vs in v): continue
    for p in range(2, len(value_sets)):
      for c in itertools.combinations(v, p):
        if len(set.union(*c)) < p: #any p of the sets together have less than p values
          break
      else: continue
      break
    else: return sum(comb), set(comb)
def max_sum_efficient(value_sets):
  for comb in itertools.combinations(sorted(set.union(*value_sets), reverse=True), len(value_sets)):
    v = list(vs.intersection(comb) for vs in value_sets)
    #now if any 1 has 0 values continue
    if any(len(vs) == 0 for vs in v): continue
    for p in range(2, len(value_sets)):
      for c in itertools.combinations(v, p):
        if len(set.union(*c)) < p: #any p of the sets together have less than p values
          break
      else: continue
      break
    else: return sum(comb), set(comb)
def add_to_first(x, tup): return (x + tup[0], tup[1]) if not tup is None else None
def min_sum(value_sets, used_values):
  if len(value_sets) == 0: return (0, used_values)
  rem = value_sets[0] - used_values
  res = tuple(filter(lambda y: not y is None, (add_to_first(x, min_sum(value_sets[1:], used_values | {x})) for x in rem)))
  return min(res) if len(res) != 0 else None
def get_all_mutex_digit_sum_group_sets(total, value_sets, used_values):
  def get_all_mutex_digit_sum_group_inner(total, value_sets, used_values):
    if total <= 0: return []
    if all((len(x)==0 for x in value_sets[1:])) and len(value_sets[0]) == 1: return [(total,)] if total in value_sets[0][0][0] and not total in used_values[0] else []
    if len(value_sets[0]) == 0: return get_all_mutex_digit_sum_group_inner(total, value_sets[1:], used_values[1:])
    l = []
    for y in value_sets[0][0][0]:
      if not y in used_values[0]:
        l.extend([(y, *x) for x in get_all_mutex_digit_sum_group_inner(total - y, [value_sets[0][1:]] + value_sets[1:], [x | set((y,)) if i == 0 or i in value_sets[0][0][1] else x for i, x in enumerate(used_values)])])
    return l
  return get_all_mutex_digit_sum_group_inner(total, value_sets, used_values)
def get_all_mutex_digit_sum_group_sets_balancing(total, value_sets, used_values, used_dict, balance_limit):
  def get_all_mutex_digit_sum_group_inner(total, value_sets, used_values, used_dict):
    if total <= 0: return []
    if sum(1 for k, v in used_dict.items() if v == 2) > balance_limit: return []
    if all((len(x)==0 for x in value_sets[1:])) and len(value_sets[0]) == 1: return [(total,)] if total in value_sets[0][0][0] and not total in used_values[0] else []
    if len(value_sets[0]) == 0: return get_all_mutex_digit_sum_group_inner(total, value_sets[1:], used_values[1:], used_dict)
    l = []
    for y in value_sets[0][0][0]:
      if not y in used_values[0]:
        l.extend([(y, *x) for x in get_all_mutex_digit_sum_group_inner(total - y, [value_sets[0][1:]] + value_sets[1:], [x | set((y,)) if i == 0 or i in value_sets[0][0][1] else x for i, x in enumerate(used_values)], {k:(v+1 if k == y else v) for k, v in used_dict.items()} if y in used_dict else {y:1, **used_dict})])
    return l
  return get_all_mutex_digit_sum_group_inner(total, value_sets, used_values, used_dict)
def has_mutex_digit_sum(total, value_sets, used_values):
  #get_all_mutex_digit_sum_sets(43, [[x for x in range(2, 10)] for _ in range(7)], set((1,)))
  def has_mutex_digit_sum_inner(total, value_sets, used_values):
    if len(value_sets) == 1: return True if total in value_sets[0] and not total in used_values else False
    if total <= 0: return False
    for y in value_sets[0]:
      if not y in used_values:
        if has_mutex_digit_sum_inner(total - y, value_sets[1:], used_values | set((y,))): return True
    return False
  return has_mutex_digit_sum_inner(total, value_sets, used_values)
def exclude_jigsaw_rule_gen(jigsaw):
  def exclude_jigsaw_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    #contiguous combinations of rows/columns summation rules?
    return []
  return exclude_jigsaw_rule
def set_of_tuples_to_tuple_of_sets(soft):
  if len(soft) == 0: return ()
  res = [set() for _ in next(iter(soft))]
  for t in soft:
    for i, x in enumerate(t): res[i].add(x)
  return res
def get_mutex_cells(cells, cell_visibility_rules, l): #now supporting non-transitive visibility groupings
  vis = [cell_visibility(x[0], x[1], l, cell_visibility_rules) for x in cells]
  for x, c in enumerate(cells):
    if x == 0:
      groups = [set((c,))]
      continue
    g = list(filter(lambda z: len(z.intersection(vis[x])) != 0, groups))
    if len(g) == 0:
      groups.append(set((c,)))
    else:
      for gx in g:
        if len(gx.intersection(vis[x])) == len(gx):
          gx.add(c)
      for gx in g:
        if len(gx.intersection(vis[x])) != len(gx):
          newgroup = gx.intersection(vis[x]) | set((c,))
          for gr in groups:
            if gr.issubset(newgroup): groups.remove(gr)
          if not any(newgroup.issubset(gr) for gr in groups): groups.append(newgroup)
  return topological_sort_groups(sorted(groups, key=lambda g: len(g), reverse=True))
#https://en.wikipedia.org/wiki/Topological_sorting
def topological_sort_groups(groups): #dont need to care about circular dependencies
  unmarked, marked, tempmark, res = groups.copy(), [], [], []
  def visit(g, res):
    if g in marked: return
    if g in tempmark: return #cycle
    tempmark.append(g)
    unmarked.remove(g)
    for gn in groups:
      if len(g.intersection(gn)) != 0:
        visit(gn, res)
    tempmark.remove(g)
    marked.append(g)
    res.insert(0, g)
  while len(unmarked) != 0:
    visit(unmarked[0], res)
  return res
def get_mutex_groups_min_dependent(groups): #this is a graph problem...
  groupcand = [g.copy() for g in groups]
  #must first define a correct dependency ordering where dependents immediately go after but requires topoligical sorting
  for i, g in enumerate(groupcand): #must enumerate in dependency order...
    dependents = []
    for p in g:
      forwards = []
      for j in range(i+1, len(groupcand)):
        if p in groupcand[j]:
          forwards.append(j-i) #using relative offset for recursion
          groupcand[j].remove(p)
      dependents.append((p, forwards))
    groupcand[i] = dependents
  return groupcand
def exclude_killer_rule_gen(killer): #digits unique in a cage
  exclude_jigsaw_rule = exclude_jigsaw_rule_gen([y for _, y in killer])
  def exclude_killer_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    tot = sum(z for z in value_set)
    for x, y in killer:
      print(x, y)
      exclude = []
      groups = list(get_mutex_cells(y, cell_visibility_rules, l))
      groupcand = get_mutex_groups_min_dependent(groups)
      sets = []
      renban = x[0] if isinstance(x, tuple) else False
      for xp in x[1] if isinstance(x, tuple) else [x]:
        #if xp > tot:
        #  sets.extend(get_all_mutex_digit_sum_group_sets_balancing(xp, [[(rem[j[0]][j[1]], f) for j, f in g] for g in groupcand], [set() for _ in groupcand], dict(), len(y) % l))
        #else:
        sets.extend(get_all_mutex_digit_sum_group_sets(xp, [[(rem[j[0]][j[1]], f) for j, f in g] for g in groupcand], [set() for _ in groupcand]))
      if renban: sets = list(filter(lambda z: max(z) - min(z) == len(y) - 1, sets))
      elif not renban and isinstance(x, tuple): sets = list(filter(lambda z: max(z) - min(z) != len(y) - 1, sets))
      yidxdict = {pt:i for i, pt in enumerate([pt for g in groupcand for pt, _ in g])}
      for i, g in enumerate(groups):
        if len(g) > (len(value_set) >> 1): #if more than half of a group, can use subtractive logic
          for regions in mutex_rules:
            for region in regions:
              if g.issubset(region): #can only use groups that were not forward referenced but prior condition should handle that
                pts = region - g
                excsets = []
                for z in range(min([sum(z[yidxdict[pt]] for pt in g) for z in sets]), max([sum(z[yidxdict[pt]] for pt in g) for z in sets])+1):
                  excsets.extend(get_all_mutex_digit_sum_group_sets(tot-z, [[(rem[f[0]][f[1]], []) for f in pts]], [set()]))
                possibles = [set() for _ in pts]
                for z in excsets:
                  for k, q in enumerate(z): possibles[k].add(q)
                for k, f in enumerate(pts):
                  for val in rem[f[0]][f[1]].difference(possibles[k]):
                    exclude.append((f[0], f[1], val))
      possibles = [set() for _ in y]
      for z in sets:
        for i, q in enumerate(z): possibles[i].add(q)
      for j in yidxdict:
        for val in rem[j[0]][j[1]].difference(possibles[yidxdict[j]]):
          exclude.append((j[0], j[1], val))
      #for i in range(len(y)):
      #  if len(rem[y[i][0]][y[i][1]]) == 1: continue
      #  for val in rem[y[i][0]][y[i][1]]:
      #    if not has_mutex_digit_sum(x - val, [rem[y[j][0]][y[j][1]] for j in range(len(y)) if j != i], set((val,))): #len(get_all_mutex_digit_sum_sets()) == 0
      #      exclude.append((y[i][0], y[i][1], val))
      if len(exclude) != 0:
        possible.append((exclude, KILLER_CAGE_RULE, (x, y)))
        return possible
      """
      common = frozenset.intersection(*(frozenset(s) for s in sets))
      for c in common:
        commonpts = [pt for pt in y if c in rem[pt[0]][pt[1]]]
        for regions in mutex_rules:
          for region in regions:
            if len(region.intersection(commonpts)) == len(commonpts):
              exclude = []
              for cell in region.difference(commonpts):
                if c in rem[cell[0]][cell[1]]:
                  exclude.append((cell[0], cell[1], c))
              if len(exclude) != 0: possible.append((exclude, KILLER_CAGE_RULE, ()))      
      """
      #yidxdict = {pt:i for i, pt in enumerate(y)}
      #unorthodox locked candidates
      for regions in mutex_rules: #because of potential repeated values, more general to find common values per region than globally
        for region in regions:
          commonpts = region.intersection(y)
          yidxs = [yidxdict[pt] for pt in commonpts]
          common = frozenset.intersection(*(frozenset(s[i] for i in yidxs) for s in sets))
          for c in common:
            exclude = []
            for cell in region.difference(commonpts):
              if c in rem[cell[0]][cell[1]]:
                exclude.append((cell[0], cell[1], c))
            if len(exclude) != 0: possible.append((exclude, KILLER_CAGE_RULE, (x, y)))
      #can exclude any sets where a double interferes with a 2 visible triples, a triple interferes with a 2/3 visible quadruples, or 3 quintuples, a quadruple interferes with a 4 quintuples
      if len(y) == 2: #this needs much better generalization, could start tracking multi-cell constraints above...
        exclude = []
        #sets = get_all_mutex_digit_sum_group_sets(x, [[rem[j[0]][j[1]] for j in g] for g in groups], set())
        #sets = get_all_mutex_digit_sum_sets(x, [rem[y[j][0]][y[j][1]] for j in range(len(y))], set())
        removesets = set()
        for regions in mutex_rules:
          for region in regions:
            intsct = region.intersection(y)
            if len(intsct) >= 2:
              for z in sets:
                vals = frozenset((z[yidxdict[p]] for p in intsct))
                found = tuple(rem[p[0]][p[1]] for p in region.difference(y) if len(vals.intersection(rem[p[0]][p[1]])) != 0)
                multiples = set.union(*found) if len(found) != 0 else set()
                if len(multiples) - len(vals.intersection(multiples)) < len(found):
                  removesets.add(z)
        exc = set_of_tuples_to_tuple_of_sets(removesets)
        filt = set_of_tuples_to_tuple_of_sets(tuple(filter(lambda s: not s in removesets, sets)))
        for i in range(len(exc)):
          for z in exc[i] - filt[i]:
            exclude.append((y[i][0], y[i][1], z))
        if len(exclude) != 0: possible.append((exclude, KILLER_CAGE_RULE, (x, y)))
            
      #then consolidate exclusions on digits
      """
      if len(y) == l-1:
        for z in y:
          if tot - x in rem[z[0]][z[1]]:
            exclude.append((z[0], z[1], tot - x))
      else:        
        for r in value_set:
          if x - r <= 0 or sum(z for z in range(1, len(y))) > x - r:
            for z in y:
              if r in rem[z[0]][z[1]]:
                exclude.append((z[0], z[1], r))
          if sum(z for z in range(l, l - len(y) + 1, -1)) < x - r:
            for z in y:
              if r in rem[z[0]][z[1]]:
                exclude.append((z[0], z[1], r))
        """ """
        for r in range(1, l - len(y) + 1 + 1):
          if sum(range(r, r + len(y))) > x:
            print(r, r + len(y), r + len(y) - 1, x, y)
            for q in filter(lambda v: v >= r + len(y) - 1, value_set):
              for z in y:
                if q in rem[z[0]][z[1]]:
                  exclude.append((z[0], z[1], q))
            break"""
    return possible
  return exclude_killer_rule
def exclude_cage_hidden_tuple_rule_gen(cages):
  def exclude_cage_hidden_tuple_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    for c in cages:
      exclude = []
      vals = set.union(*(rem[p[0]][p[1]] for p in c))
      if len(vals) == len(c):
        for y in vals:
          a = tuple(cell_visibility(i, j, l, cell_visibility_rules) for i, j in c if y in rem[i][j])
          visible = frozenset.intersection(*a) if len(a) != 0 else frozenset()
          for i, j in visible:
            if y in rem[i][j]: exclude.append((i, j, y))
      if len(exclude) != 0: possible.append((exclude, HIDDEN_CAGE_TUPLE, (c)))
    return possible
  return exclude_cage_hidden_tuple_rule
def exclude_cage_mirror_rule_gen(cages):
  def exclude_cage_mirror_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    exclude = []
    for c in cages:
      for i in range(len(c)):
        a = tuple(cell_visibility(x, y, l, cell_visibility_rules) for x, y in c[:i] + c[i+1:]) #if len(rem[x][y]) != 1
        visible = frozenset.intersection(*a) if len(a) != 0 else frozenset()
        visible = {x for x in visible if len(rem[x[0]][x[1]]) != 1}
        if len(visible) == 1 and len(visible.intersection(cell_visibility(c[i][0], c[i][1], l, cell_visibility_rules))) == 0:
          p = next(iter(visible))
          for y in rem[c[i][0]][c[i][1]] - rem[p[0]][p[1]]:
            exclude.append((c[i][0], c[i][1], y))
            print(visible, c, y)
    if len(exclude) != 0: possible.append((exclude, MIRROR_CAGE_CELL, ()))
    return possible
  return exclude_cage_mirror_rule
def get_min_max_sums(value_set):
  mins, maxes = [], [] #mins nearest rounded down, maxes nearest rounded up
  for x in sorted(value_set):
    if len(mins) == 0: mins.append(x)
    else: mins.append(mins[-1] + x)
  for x in reversed(sorted(value_set)):
    if len(maxes) == 0: maxes.append(x)
    else: maxes.append(maxes[-1] + x)
  return mins, maxes
def exclude_sandwich_rule_gen(sandwiches):
  def exclude_sandwich_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    tot = sum(value_set)
    mins, maxes = get_min_max_sums(value_set) #mins nearest rounded down, maxes nearest rounded up
    sandwichset = set((min(value_set), max(value_set))) #1 and 9
    for x in sandwiches:
      remtot = tot - sum(sandwichset) - x[0]
      maxlen = len(tuple(itertools.takewhile(lambda i: i <= x[0], mins)))
      minlen = len(tuple(itertools.takewhile(lambda i: i <= x[0], maxes)))
      if x[0] != 0 and maxes[minlen-1] != x[0]: minlen += 1
      contig = tuple(sorted(x[1]))
      possibles = [set() for _ in range(len(x[1]))]
      for z in range(minlen, maxlen+1, 1):
        #first pivot to 1s and 9s that are spaced correctly
        #then check inside the sandwich combinations along with all of its outside the sandwich combinations
        #for each combination that works add to an inclusive set
        for p in range(1, len(x[1]) - z):
          p1, p2 = contig[p-1], contig[p + z]
          v1, v2 = rem[p1[0]][p1[1]].intersection(sandwichset), rem[p2[0]][p2[1]].intersection(sandwichset)
          if len(v1) == 0 or len(v2) == 0: continue
          elif len(v1) == 1 and len(v2) == 1 and v1 == v2: continue
          elif len(v1) == 1 and len(v2) == 2: v2 -= v1
          elif len(v1) == 2 and len(v2) == 1: v1 -= v2
          if x[0] != 0:
            res = get_all_mutex_digit_sum_sets(x[0], [rem[contig[i][0]][contig[i][1]] for i in range(p, p+z)], sandwichset)
          else: res = []
          if z != len(x[1]) - 1 - 1: #x[0] != tot - sum(sandwichset) and 
            remres = get_all_mutex_digit_sum_sets(remtot, [rem[contig[i][0]][contig[i][1]] for i in range(len(x[1])) if i < p-1 or i > p + z], sandwichset)
          else: remres = []
          if x[0] == 0 and len(remres) != 0 or z == len(x[1]) - 1 - 1 and len(res) != 0 or len(res) != 0 and len(remres) != 0:
            possibles[p-1] |= v1
            possibles[p+z] |= v2
            for y in res:
              for i in range(p, p+z):
                possibles[i].add(y[i - p])
            for y in remres:
              for i in range(len(x[1])):
                if i < p-1: possibles[i].add(y[i])
                elif i > p + z: possibles[i].add(y[i - z - 1 - 1]) #p - 1 + i - p - z - 1
      #exclude all combinations that are not in the inclusive set
      exclude = []
      for i in range(len(x[1])):
        for y in rem[contig[i][0]][contig[i][1]].difference(possibles[i]):
          exclude.append((contig[i][0], contig[i][1], y))
      if len(exclude) != 0: possible.append((exclude, SANDWICH_RULE, ()))
    return possible
  return exclude_sandwich_rule
def exclude_battlefield_rule_gen(fields):
  def exclude_battlefield_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    tot, mx = sum(value_set), max(value_set)
    fieldsums = []
    for val, points in fields:
      fieldsums.append(set())
      contig = tuple(sorted(points))
      start, stop = contig[0], contig[-1]
      possibles = [set() for _ in range(l)]
      for s, p in itertools.product(rem[start[0]][start[1]], rem[stop[0]][stop[1]]):
        if s == p: continue
        if s + p == l: #could have a 0 clue in some puzzle
          if val == 0:
            possibles[0].add(s)
            possibles[-1].add(p)
            fieldsums[-1].add(s + p)
          continue
        startidx, endidx = (s - (1 if s == mx else 0), l-p + (1 if p == mx else 0)) if s + p <= l else (l-p + (1 if p == mx else 0), s - (1 if s == mx else 0)) #gap vs. overlap
        respts = [rem[pt[0]][pt[1]] for pt in contig[startidx:endidx]]
        res = get_all_mutex_digit_sum_sets(val - (0 if s != mx and p != mx else (s if p == mx else p)), respts, set((s, p))) if len(respts) != 0 else []
        rempts = [rem[pt[0]][pt[1]] for pt in contig[1:startidx] + contig[endidx:l-1]]
        #now an overlap can actually include the start or end value if one of them is a 9
        if len(rempts) != 0 and tot - (s + p if s != mx and p != mx else (p if p == mx else s)) - val <= 0:
          continue
        remres = get_all_mutex_digit_sum_sets(tot - (s + p if s != mx and p != mx else (p if p == mx else s)) - val, rempts, set((s, p))) if len(rempts) != 0 else []
        if (len(res) != 0 or len(respts) == 0) and (len(remres) != 0 or len(rempts) == 0):
          possibles[0].add(s)
          possibles[-1].add(p)
          fieldsums[-1].add(s + p)
          for y in res:
            for i in range(startidx, endidx):
              possibles[i].add(y[i-startidx])
          for y in remres:
            for i in range(len(y)):
              possibles[i + (1 if i < startidx - 1 else endidx - startidx + 1)].add(y[i])
      exclude = []
      for i, pt in enumerate(contig):
        if val == 0 and not i in (0, len(contig)-1): continue
        for y in rem[pt[0]][pt[1]].difference(possibles[i]):
          exclude.append((pt[0], pt[1], y))
      if len(exclude) != 0: possible.append((exclude, BATTLEFIELD_RULE, ()))
    for indexes in get_mutex_cages([y for _, y in fields], l)[l-1]: #length l mutually exclusive regions
      fieldset = [fields[i] for i in indexes]
      fieldsetsums = [fieldsums[i] for i in indexes]
      allposs = [[set(), set()] for _ in fieldset]
      for sums in get_all_mutex_digit_group_sum(tot << 1, [1 for _ in range(l)], fieldsetsums):
        for i, (val, points) in enumerate(fieldset):
          contig = tuple(sorted(points))
          start, stop = contig[0], contig[-1]
          for s, p in itertools.product(rem[start[0]][start[1]], rem[stop[0]][stop[1]]):
            if s == p: continue
            if s + p == sums[i]:
              allposs[i][0].add(s); allposs[i][-1].add(p)
      exclude = []
      for i, (val, points) in enumerate(fieldset):
        contig = tuple(sorted(points))
        for j, pt in enumerate((contig[0], contig[-1])):
          for y in rem[pt[0]][pt[1]].difference(allposs[i][j]):
            exclude.append((pt[0], pt[1], y))
      if len(exclude) != 0: possible.append((exclude, BATTLEFIELD_RULE, ()))
    return possible
  return exclude_battlefield_rule
def exclude_renban_rule_gen(cages):
  def exclude_renban_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    for cagesets in cages:
      allprods, allcommon, allposs = [], [], []
      for renban, cells in cagesets:
        prods = []
        common = value_set.copy()
        for prod in itertools.product(*[rem[pt[0]][pt[1]] for pt in cells]): #such complexity is not required
          if len(set(prod)) == len(cells) and not (renban ^ (max(prod) - min(prod) + 1 == len(cells))):
            common = common.intersection(prod)
            prods.append(prod)
        allprods.append(prods); allcommon.append(common)
      if len(allprods) != 1: allprods = [list(filter(lambda seq: all(any(len(set(seq).intersection(y)) == 0 for y in z) for z in allprods[:i] + allprods[i+1:]), x)) for i, x in enumerate(allprods)]
      for k, prods in enumerate(allprods):
        possibles = [set() for _ in cagesets[k][1]]
        for prod in prods:
          for i, x in enumerate(prod):
            possibles[i].add(x)
        allposs.append(possibles)
      exclude = []
      for k, (_, cells) in enumerate(cagesets):
        for i, pt in enumerate(cells):
          for y in rem[pt[0]][pt[1]].difference(allposs[k][i]):
            exclude.append((pt[0], pt[1], y))
      if len(exclude) != 0: possible.append((exclude, RENBAN_RULE, ()))      
      for k, (_, cells) in enumerate(cagesets):
        for c in allcommon[k]:
          commonpts = [pt for pt in cells if c in rem[pt[0]][pt[1]]]
          for regions in mutex_rules:
            for region in regions:
              if len(region.intersection(commonpts)) == len(commonpts):
                exclude = []
                for cell in region.difference(commonpts):
                  if c in rem[cell[0]][cell[1]]:
                    exclude.append((cell[0], cell[1], c))
                if len(exclude) != 0: possible.append((exclude, RENBAN_RULE, ()))
    return possible
  return exclude_renban_rule
def exclude_in_order_rule_gen(clues):
  def exclude_in_order_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    for c, points in clues:
      contig = tuple(sorted(points))
      exclude = []
      for i, x in enumerate(c):
        for j in range(i):
          if x in rem[contig[j][0]][contig[j][1]]: exclude.append((contig[j][0], contig[j][1], x))
        for j in range(l - len(c) + i+1, l):
          if x in rem[contig[j][0]][contig[j][1]]: exclude.append((contig[j][0], contig[j][1], x))
      if len(exclude) != 0: possible.append((exclude, IN_ORDER_RULE, ()))
      exclude = []
      j = 0
      for i, x in enumerate(c):
        while not x in rem[contig[j][0]][contig[j][1]]:
          for y in c[i+1:]:
            if y in rem[contig[j][0]][contig[j][1]]: exclude.append((contig[j][0], contig[j][1], y))
          j += 1
        j += 1
      j = l - 1
      for i, x in enumerate(reversed(c)):
        while not x in rem[contig[j][0]][contig[j][1]]:
          for y in c[:len(c)-i-1]:
            if y in rem[contig[j][0]][contig[j][1]]: exclude.append((contig[j][0], contig[j][1], y))
          j -= 1
        j -= 1
      if len(exclude) != 0: possible.append((exclude, IN_ORDER_RULE, ()))
      exclude = []
      for i, x in enumerate(c):
        for k in [j for j in range(l) if x in rem[contig[j][0]][contig[j][1]]][1:]:
          m = k + 1
          for n in c[i+1:]:
            while m != l and not n in rem[contig[m][0]][contig[m][1]]: m += 1
            if m == l:
              exclude.append((contig[k][0], contig[k][1], x))
              break
      if len(exclude) != 0: possible.append((exclude, IN_ORDER_RULE, ()))
      exclude = []
      for i, x in enumerate(reversed(c)):
        for k in [j for j in range(l) if x in rem[contig[j][0]][contig[j][1]]][:-1]:
          m = k - 1
          for n in reversed(c[:len(c)-i-1]):
            while m != -1 and not n in rem[contig[m][0]][contig[m][1]]: m -= 1
            if m == -1:
              exclude.append((contig[k][0], contig[k][1], x))
              break
      if len(exclude) != 0: possible.append((exclude, IN_ORDER_RULE, ()))
    return possible
  return exclude_in_order_rule
def exclude_orthagonal(rem, mutex_rules, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  for i in range(l):
    for j in range(l):
      if len(rem[i][j]) == 1:
        exclusions = set((next(iter(rem[i][j])) - 1, next(iter(rem[i][j])) + 1))
      elif len(rem[i][j]) == 2:
        exclusions = set.intersection({x-1 for x in rem[i][j]}, {x+1 for x in rem[i][j]})
      else: exclusions = set()
      if len(exclusions) == 0: continue
      exclude = []
      for k in frozenset.union(*orthagonal_points(i, j, l)):
        for val in rem[k[0]][k[1]].intersection(exclusions):
          exclude.append((k[0], k[1], val))
      if len(exclude) != 0: possible.append((exclude, ORTHAGONAL_NEIGHBOR, (i, j, exclusions)))
  return possible
def exclude_even_odd_gen(cells, is_even):
  def exclude_even_odd(rem, mutex_rules, cell_visibility_rules, value_set):
    possible = []
    for p in cells:
      exclude = []
      for y in rem[p[0]][p[1]]:
        if (y & 1 == 0) != is_even:
          exclude.append((p[0], p[1], y))
      if len(exclude) != 0: possible.append((exclude, EVEN_ODD, ()))
    return possible
  return exclude_even_odd
def exclude_small_big_gen(cells, is_small):
  def exclude_small_big(rem, mutex_rules, cell_visibility_rules, value_set):
    possible = []
    mid = list(sorted(value_set))[len(value_set) >> 1]
    for p in cells:
      exclude = []
      for y in rem[p[0]][p[1]]:
        if (y < mid) != is_small:
          exclude.append((p[0], p[1], y))
      if len(exclude) != 0: possible.append((exclude, SMALL_BIG, ()))
    return possible
  return exclude_small_big
def exclude_symmetry(rem, mutex_rules, cell_visibility_rules, value_set): #only 180 degree symmetry currently, should generalize for mirrors and 90/270 rotations
  l, possible = len(rem), []
  symsum = min(value_set) + max(value_set)
  for i in range(l): #only need to do the upper right from the diagonal
    for j in range(l):
      if i + j >= l: continue
      if i + j == l - 1 and i > (l >> 1): continue #only need to do half of the diagonal and middle cell
      if i + j == l - 1 and i == j:
        values = set(filter(lambda v: symsum - v == v, rem[i][j]))
      else:
        values = rem[i][j].intersection({symsum - v for v in rem[l-1-i][l-1-j]})
      exclude = []
      for y in rem[i][j] - values:
        exclude.append((i, j, y))
      if i + j != l - 1 or i != j:
        symvalues = rem[l-1-i][l-1-j].intersection({symsum - v for v in rem[i][j]})
        for y in rem[l-1-i][l-1-j] - symvalues:
          exclude.append((l-1-i, l-1-j, y))
      if len(exclude) != 0: possible.append((exclude, SYMMETRY_RULE, ()))
  return possible
def exclude_surplus_gen(cages):
  def surplus_mutex_rules(l):
    return [y for y in cages.values() if len(y) >= l]
  def exclude_surplus(rem, mutex_rules, cell_visibility_rules, value_set):
    possible = []
    possible.extend(hidden_single(rem, mutex_rules, cell_visibility_rules, value_set, surplus_mutex_rules))
    possible.extend(locked_candidates(rem, mutex_rules, cell_visibility_rules, value_set, dynamic_mutex_rules=surplus_mutex_rules))
    return possible
  return exclude_surplus
def exclude_queen_rule_gen(queen_vals):
  def queen_visibility(i, j, rem, y):
    l = len(rem)
    if y in queen_vals:
      return frozenset.union(*bishop_rule_points(i, j, l))
    return frozenset()
  def exclude_queen_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    possible = []
    possible.extend(naked_single(rem, mutex_rules, cell_visibility_rules, value_set, queen_visibility))
    possible.extend(locked_candidates(rem, mutex_rules, cell_visibility_rules, value_set, queen_visibility))
    possible.extend(x_chain(rem, mutex_rules, cell_visibility_rules, value_set, 4, queen_visibility))
    return possible
  return exclude_queen_rule
def exclude_jousting_knights_rule_gen(exclusions, exc_digits):
  def knight_exclusion_visibility(i, j, rem, y):
    points = frozenset.union(*knight_rule_points(i, j, len(rem)))
    if (i, j) in exclusions:
      return points
    else:
      partner = [p for p in points if len(rem[p[0]][p[1]]) == 1 and y in rem[p[0]][p[1]]]
      if len(partner) != 0:
        return frozenset.union(points, *knight_rule_points(partner[0][0], partner[0][1], len(rem))) - frozenset(((i, j), *partner))
    return frozenset()
  def exclude_jousting_knights_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    for i in range(l):
      for j in range(l):
        if (i, j) in exclusions: continue
        visible = cell_visibility(i, j, l, cell_visibility_rules)
        exclude = []
        if len(rem[i][j]) == 1:
          y = next(iter(rem[i][j]))
          if y in exc_digits: continue
          points = [p for p in frozenset.union(*knight_rule_points(i, j, l)) if y in rem[p[0]][p[1]] and not p in visible and not p in exclusions]
          if len(points) == 1:
            for j in rem[points[0][0]][points[0][1]]:
              if j == y: continue
              exclude.append((points[0][0], points[0][1], j))
        else:
          for y in rem[i][j] - exc_digits:
            points = [p for p in frozenset.union(*knight_rule_points(i, j, l)) if y in rem[p[0]][p[1]] and not p in visible and not p in exclusions]
            if len(points) == 0:
              exclude.append((i, j, y))
            elif len(points) == 1:
              pt = points[0]
              allvis = frozenset.union(visible, frozenset(points), cell_visibility(pt[0], pt[1], l, cell_visibility_rules), frozenset([p for p in frozenset.union(*knight_rule_points(pt[0], pt[1], l)) if y in rem[p[0]][p[1]] and not p in visible and not p in exclusions]))
              for regions in mutex_rules:
                for region in regions:
                  if pt in region or (i, j) in region: continue
                  region = frozenset(q for q in region if y in rem[q[0]][q[1]])
                  if len(region.intersection(allvis)) == len(region):
                    exclude.append((i, j, y))
                    break
                else: continue
                break
        if len(exclude) != 0: possible.append((exclude, JOUSTING_KNIGHTS, ()))
    possible.extend(naked_single(rem, mutex_rules, cell_visibility_rules, value_set, knight_exclusion_visibility))
    possible.extend(locked_candidates(rem, mutex_rules, cell_visibility_rules, value_set, knight_exclusion_visibility))
    #possible.extend(x_chain(rem, mutex_rules, cell_visibility_rules, value_set, 4, knight_exclusion_visibility))
    return possible
  return exclude_jousting_knights_rule
def get_fibonacci_num(n, x = (1, 1)):
  #(((1+sqrt(5))/2)^n-((1-sqrt(5))/2)^n)/sqrt(5)
  #((1+sqrt(5))^n-(1-sqrt(5))^n)/(2^n*sqrt(5) - sqrt(5) is just a place held by a variable e.g. x since it must cancel
  if n <= 0: raise ValueError
  a, b, c, d = 1, 1, 1, -1
  e, f, g, h = 1, 0, 1, 0
  exp = n
  while exp > 1:
    if (exp & 1) != 0:
      e, f = a * e + b * f * 5, a * f + e * b
      g, h = c * g + d * h * 5, c * h + g * d
    a, b = a * a + b * b * 5, (a * b) << 1
    c, d = c * c + d * d * 5, (c * d) << 1
    exp >>= 1
  a, b = a * e + b * f * 5, a * f + e * b
  c, d = c * g + d * h * 5, c * h + g * d
  #if a != c: raise ValueError
  if x != (1, 1): return ((b - d) >> n) * x[0] + ((a - b) >> n) * x[1]
  return (b - d) >> n
def get_fibonacci_num_matrix(n, x = (1, 1)):
  import numpy as np
  if n <= 0: raise ValueError
  mat = np.array(((1, 1), (1, 0)))
  y = np.array(((1, 0), (0, 1)))
  while n > 1:
    if (n & 1) != 0: y = mat.dot(y)
    mat = mat.dot(mat)
    n >>= 1
  if x != (1, 1): return mat.dot(y).dot(np.array(x))[1]
  return mat.dot(y)[0, 1]
def gen_fibonacci_seqs(set1, set2, max_digits):
  seqs = []
  for x in set1:
    for y in set2:
      seq, num1, num2 = [x], x, y
      while True:
        digits, val = [], num2
        while val != 0:
          digits.append(val % 10)
          val //= 10
        if 0 in digits: break
        seq.extend(reversed(digits))
        if len(seq) == max_digits: seqs.append(seq)
        if len(seq) >= max_digits: break
        num1, num2 = num2, num1 + num2
  return seqs
def gen_fibonacci_sets_seqs(sets):
  seqs = []
  for x in sets[0]:
    for y in sets[1]:
      seq, num1, num2 = [x], x, y
      while True:
        digits, val = [], num2
        while val != 0:
          digits.append(val % 10)
          val //= 10
        if len(sets) < len(seq) + len(digits): break
        if any(not d in sets[len(seq)+i] for i, d in enumerate(reversed(digits))): break
        seq.extend(reversed(digits))
        if len(seq) == len(sets): seqs.append(seq); break
        num1, num2 = num2, num1 + num2
  return seqs
def exclude_fibonacci_rule_gen(cages):
  def exclude_fibonacci_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    for start, cage in cages:
      vals = [rem[pt[0]][pt[1]] for pt in cage]
      allseqs = []
      cageidxdict = {c:i for i, c in enumerate(cage)}
      for j in (0,) if start else range(len(cage)):
        seqs = gen_fibonacci_sets_seqs(vals[j:] + vals[:j])
        for i, pt in enumerate(cage):
          intsct = cell_visibility(pt[0], pt[1], l, cell_visibility_rules).intersection(cage)
          seqs = list(filter(lambda s: all(s[i] != s[x] for x in [cageidxdict[y] for y in intsct]), seqs))
        allseqs.extend([s[-j:] + s[:-j] for s in seqs])
      exclude = []
      for i, pt in enumerate(cage):
        for y in rem[pt[0]][pt[1]] - set(s[i] for s in allseqs):
          exclude.append((pt[0], pt[1], y))
      if len(exclude) != 0: possible.append((exclude, FIBONACCI_RULE, ()))
    return possible
  return exclude_fibonacci_rule
def exclude_double_rule_gen(doubles):
  def exclude_double_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    possible = []
    for d in doubles:
      exclude = []
      for i in range(len(d)):
        poss = {x << 1 for x in rem[d[1-i][0]][d[1-i][1]] if x << 1 in value_set} | {x for x in rem[d[i][0]][d[i][1]] if x << 1 in value_set}
        for y in rem[d[i][0]][d[i][1]] - poss:
          exclude.append((d[i][0], d[i][1], y))
      if len(exclude) != 0: possible.append((exclude, DOUBLE_RULE, ()))
    return possible
  return exclude_double_rule
def manhat_dist(taxi1, taxi2): return abs(taxi1[0] - taxi2[0]) + abs(taxi1[1] - taxi2[1])
def manhattan_cells(i, j, l, dist):
  return filter_bounds_points(l, frozenset((i + x, j + dist - abs(x)) for x in range(-dist, dist+1)) | frozenset((i + x, j - dist + abs(x)) for x in range(-dist, dist+1)))
def exclude_manhattan_rule_gen(taxis, max_subset):
  def manhattan_exclusion_visibility(i, j, rem, y):
    if not (i, j) in taxis:
      #return frozenset() if len(rem[i][j]) != 1 else frozenset(manhattan_cells(i, j, len(rem), next(iter(rem[i][j]))))
      return frozenset(manhattan_cells(i, j, len(rem), y))
    elif len(rem[i][j]) == 1:
      return frozenset(manhattan_cells(i, j, len(rem), y)).difference(taxis)
    return frozenset()
  def exclude_manhattan_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    possible.extend(naked_single(rem, mutex_rules, cell_visibility_rules, value_set, manhattan_exclusion_visibility))
    possible.extend(locked_candidates(rem, mutex_rules, cell_visibility_rules, value_set, manhattan_exclusion_visibility))
    """
    exclude = []
    for i in range(l): #white cells cannot be a taxi move apart
      for j in range(l):
        if (i, j) in taxis or len(rem[i][j]) != 1: continue
        x = next(iter(rem[i][j]))
        for c in manhattan_cells(i, j, l, x):
          if x in rem[c[0]][c[1]]: exclude.append((c[0], c[1], x))
    if len(exclude) != 0: possible.append((exclude, MANHATTAN_RULE, ()))
    """
    exclude = []
    v = {*(manhat_dist(taxi1, taxi2) for taxi1, taxi2 in itertools.combinations(taxis, 2) if manhat_dist(taxi1, taxi2) in rem[taxi1[0]][taxi1[1]].intersection(rem[taxi2[0]][taxi2[1]]))}
    for i, taxi in enumerate(taxis): #compute possible valid taxi partners
      vis = cell_visibility(taxi[0], taxi[1], l, cell_visibility_rules)
      vals = {*(manhat_dist(taxi, t) for t in taxis if t != taxi and not t in vis and manhat_dist(taxi, t) in rem[taxi[0]][taxi[1]].intersection(rem[t[0]][t[1]]))}
      #vals = set(filter(lambda x: all(len(rem[y[0]][y[1]]) != 1 or not x in rem[y[0]][y[1]] for y in manhattan_cells(taxi[0], taxi[1], l, x)), vals))
      for y in rem[taxi[0]][taxi[1]] - vals:
        exclude.append((taxi[0], taxi[1], y))
      a = tuple(set(p) for t in taxis if t != taxi and t in vis for p in itertools.product(rem[taxi[0]][taxi[1]], rem[t[0]][t[1]]) if sum(p) < max_subset)
      if len(a) != 0: v &= set.union(*a)
    if len(exclude) != 0: possible.append((exclude, MANHATTAN_RULE, ()))
    exclude = []
    for t in taxis: #exclude based on max_subset sum
      for y in rem[t[0]][t[1]] - v:
        exclude.append((t[0], t[1], y))
    if len(exclude) != 0: possible.append((exclude, MANHATTAN_RULE, ()))
    return possible
  return exclude_manhattan_rule
def exclude_mirror_gen(mirrors):
  def exclude_mirror(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    for p, q in mirrors:
      exclude = []
      inc = rem[p[0]][p[1]].intersection(rem[q[0]][q[1]])
      for y in rem[p[0]][p[1]] - inc:
        exclude.append((p[0], p[1], y))
      for y in rem[q[0]][q[1]] - inc:
        exclude.append((q[0], q[1], y))
      if len(exclude) != 0: possible.append((exclude, MIRROR_RULE, ()))
      else: #rem[p[0]][p[1]] == rem[q[0]][q[1]]
        #cage mirror rule - technically all combinations of disconnected jigsaws could be generated but complexity is too absurd
        comb = cell_visibility(p[0], p[1], l, cell_visibility_rules).union(cell_visibility(q[0], q[1], l, cell_visibility_rules))
        for regions in mutex_rules:
          for region in regions:
            intsct = comb.intersection(region)
            if len(region - intsct) == 0 or len(intsct) == 0: continue
            pts = [pt for pt in region - intsct if len(rem[p[0]][p[1]].intersection(rem[pt[0]][pt[1]])) != 0]
            exclude = []
            if len(pts) == 1:
              #print(rem[p[0]][p[1]], pts, rem[pts[0][0]][pts[0][1]])
              for y in rem[pts[0][0]][pts[0][1]] - rem[p[0]][p[1]]:
                exclude.append((pts[0][0], pts[0][1], y))
            if len(exclude) != 0: possible.append((exclude, MIRROR_RULE, ()))
    return possible
  return exclude_mirror
def mirror_visibility_gen(mirrors, mutex_rules, cell_visibility_rules):
  cvr = mutex_regions_to_visibility(mutex_rules) + cell_visibility_rules
  def mirror_visibility(i, j, l):
    return (frozenset(x[1 if x[0] == p else 0] for p in cell_visibility(i, j, l, cvr) for x in mirrors if p in x),)
    #return (frozenset(x[1 if x[0] == (i, j) else 0] for x in mirrors if (i, j) in x),)
  return mirror_visibility
def jigsaw_points_gen(jigsaw):
  def jigsaw_points(i, j, l):
    for y in jigsaw:
      if (i, j) in y: return (frozenset(y) - frozenset(((i, j),)),)
    return (frozenset(),)
  return jigsaw_points
def killer_puzzle_gen(killer):
  return jigsaw_points_gen([y for _, y in killer])
def magic_square_rows(msc, size, gaps):
  rows = []
  for i in range(-(size >> 1), (size >> 1) + 1):
    rows.append(tuple((msc[0] + i * (gaps + 1), msc[1] + j * (gaps + 1)) for j in range(-(size >> 1), (size >> 1) + 1)))
  return rows
def magic_square_center_to_killer_cages(ms, value_set):
  l = len(value_set)
  s = isqrt(l)
  if s * s != l: return possible
  sm = sum(value_set) // s
  return [*((sm, x) for x in ms), *((sm, tuple(ms[j][i] for j in range(s))) for i in range(s)),
          (sm, tuple(ms[i][i] for i in range(s))), (sm, tuple(ms[i][s-1-i] for i in range(s)))]  
  #return ((sm, ((msc[0] - 1, msc[1] - 1), (msc[0] - 1, msc[1]), (msc[0] - 1, msc[1] + 1))),
  #        (sm, ((msc[0], msc[1] - 1), (msc[0], msc[1]), (msc[0], msc[1] + 1))),
  #        (sm, ((msc[0] + 1, msc[1] - 1), (msc[0] + 1, msc[1]), (msc[0] + 1, msc[1] + 1))),
  #        (sm, ((msc[0] - 1, msc[1] - 1), (msc[0], msc[1] - 1), (msc[0] + 1, msc[1] - 1))),
  #        (sm, ((msc[0] - 1, msc[1]), (msc[0], msc[1]), (msc[0] + 1, msc[1]))),
  #        (sm, ((msc[0] - 1, msc[1] + 1), (msc[0], msc[1] + 1), (msc[0] + 1, msc[1] + 1))),
  #        (sm, ((msc[0] - 1, msc[1] - 1), (msc[0], msc[1]), (msc[0] + 1, msc[1] + 1))),
  #        (sm, ((msc[0] - 1, msc[1] + 1), (msc[0], msc[1]), (msc[0] + 1, msc[1] - 1))))

def check_puzzle(y):
  rem, solve_path, border = solve_sudoku(y[0], y[1], y[2], y[3], y[4], y[5])
  rem, valid = check_sudoku(rem, y[1], y[4])
  #print(logical_solve_string(solve_path, y[1]))
  if not rem is None:
    y[6][0](rem) if y[5] is None else y[6][0](rem, border)
    if not valid:
      #y[6][0](brute_sudoku_depth(rem, y[2], y[3]))
      y[6][1](rem) if y[5] is None else y[6][0](rem, border)
  else: print("Bad Sudoku")

def check_puzzles(puzzles):
  for y in puzzles: check_puzzle(y)

def str_to_sudoku(s):
  l = isqrt(len(s))
  sudoku = [[[] for _ in range(l)] for _ in range(l)]
  for i in range(len(s)):
    sudoku[i // l][i % l] = int(s[i]) if s[i] != '.' else None
  return sudoku

def standard_sudoku(puzzle):
  l = len(puzzle) #(bifurcate,)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def king_sudoku(puzzle):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (king_rule_points,), (), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def knight_sudoku(puzzle):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (knight_rule_points,), (), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def jousting_knights_sudoku(puzzle, exclusions, exc_digits):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_jousting_knights_rule_gen(exclusions, exc_digits),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def queen_sudoku(puzzle, queen_vals):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_queen_rule_gen(queen_vals),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def knight_thermo_magic_square_sudoku(puzzle, thermo, magic_squares):
  l = len(puzzle)
  ms = [magic_square_rows((x[0], x[1]), x[2], x[3]) for x in magic_squares]
  killers = [magic_square_center_to_killer_cages(x, frozenset(range(1, l+1))) for x in ms]
  return (puzzle, standard_sudoku_mutex_regions(l), (knight_rule_points,), (exclude_magic_square_rule_gen(ms), *(exclude_killer_rule_gen(killer) for killer in killers), exclude_thermo_rule_gen(thermo),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def king_knight_magic_square_sudoku(puzzle, magic_squares):
  l = len(puzzle)
  ms = [magic_square_rows((x[0], x[1]), x[2], x[3]) for x in magic_squares]
  killers = [magic_square_center_to_killer_cages(x, frozenset(range(1, l+1))) for x in ms]
  return (puzzle, standard_sudoku_mutex_regions(l), (king_rule_points,knight_rule_points,), (exclude_magic_square_rule_gen(ms), *(exclude_killer_rule_gen(killer) for killer in killers),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def king_knight_orthagonal_sudoku(puzzle):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (king_rule_points,knight_rule_points,), (exclude_orthagonal,), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def jigsaw_sudoku(puzzle, jigsaw):
  l = len(puzzle)
  return (puzzle, (*standard_sudoku_singoverlap_regions(l), tuple(frozenset(x) for x in jigsaw_to_coords(jigsaw).values())), (), (), frozenset(range(1, l+1)), None,
          (lambda x: print_border(x, exc_from_border(jigsaw, set())), lambda x: print_candidate_border(x, jigsaw)))
def deficit_jigsaw_sudoku(puzzle, jigsaw):
  l = len(puzzle)
  return (puzzle, standard_sudoku_singoverlap_regions(l), (jigsaw_points_gen(jigsaw_to_coords(jigsaw).values()),), (), frozenset(range(1, l+1)), None,
          (lambda x: print_border(x, exc_from_border(jigsaw, set())), lambda x: print_candidate_border(x, jigsaw)))
def surplus_jigsaw_sudoku(puzzle, jigsaw):
  l = len(puzzle)
  return (puzzle, standard_sudoku_singoverlap_regions(l), (), (exclude_surplus_gen(jigsaw_to_coords(jigsaw)),), frozenset(range(1, l+1)), None,
          (lambda x: print_border(x, exc_from_border(jigsaw, set())), lambda x: print_candidate_border(x, jigsaw)))
def expand_row_col_top_bottom_clue(clues, condition, gen):
  return [y for x in [gen(i, 0) for i, x in enumerate(clues[0]) if condition(x)] +
                     [gen(i, 1) for i, x in enumerate(clues[1]) if condition(x)] +
                     [gen(i, 2) for i, x in enumerate(clues[2]) if condition(x)] +
                     [gen(i, 3) for i, x in enumerate(clues[3]) if condition(x)] for y in x]
def even_odd_small_big_jigsaw_sudoku(puzzle, jigsaw, clues):
  l = len(puzzle)
  def first_two_gen(i, side):
    if side == 0: return ((i, 0), (i, 1))
    elif side == 1: return ((0, i), (1, i))
    elif side == 2: return ((i, l-1), (i, l-2))
    elif side == 3: return ((l-1, i), (l-2, i))
  evens = expand_row_col_top_bottom_clue(clues, lambda x: x == 'E', first_two_gen)
  odds = expand_row_col_top_bottom_clue(clues, lambda x: x == 'O', first_two_gen)
  smalls = expand_row_col_top_bottom_clue(clues, lambda x: x == 'S', first_two_gen)
  bigs = expand_row_col_top_bottom_clue(clues, lambda x: x == 'B', first_two_gen)
  return (puzzle, (*standard_sudoku_singoverlap_regions(l), tuple(frozenset(x) for x in jigsaw_to_coords(jigsaw).values())), (), (exclude_even_odd_gen(evens, True), exclude_even_odd_gen(odds, False), exclude_small_big_gen(smalls, True), exclude_small_big_gen(bigs, False)), frozenset(range(1, l+1)), None,
          (lambda x: print_border(x, exc_from_border(jigsaw, set())), lambda x: print_candidate_border(x, jigsaw)))
def thermo_sudoku(puzzle, thermo):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_thermo_rule_gen(thermo),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def thermo_sandwich_sudoku(puzzle, sandwich_row_cols, thermo):
  l = len(puzzle)
  sandwiches = [(x, row_points(i, 0, l)) for i, x in enumerate(sandwich_row_cols[0]) if not x is None] + [(x, column_points(0, i, l)) for i, x in enumerate(sandwich_row_cols[1]) if not x is None]
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_sandwich_rule_gen(sandwiches),exclude_thermo_rule_gen(thermo),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def thermo_sudoku_symmetry(puzzle, thermo):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_thermo_rule_gen(thermo),exclude_symmetry), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))  
def arrow_sudoku(puzzle, arrows, increasing):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_arrow_rule_gen(arrows, increasing),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def partial_border_jigsaw_sudoku(puzzle, exclusions):
  l = len(puzzle)
  return (puzzle, standard_sudoku_singoverlap_regions(l), (), (), frozenset(range(1, l+1)), lambda: brute_border(exclusions, l, puzzle, standard_sudoku_singoverlap_regions(l), (), (), frozenset(range(1, l+1)))[1],
          (lambda x, border: print_border(x, exc_from_border(border, exclusions)), lambda x, border: print_candidate_border(x, exc_from_border(border, exclusions))))
def partial_border_diagonal_thermo_sudoku(puzzle, exclusions, thermo):
  l = len(puzzle)
  #exclude_inequality_rule_gen(inequalities)
  return (puzzle, (*standard_sudoku_singoverlap_regions(l), diagonal_sudoku_regions(l)), (), (exclude_thermo_rule_gen(thermo),), frozenset(range(1, l+1)), lambda: brute_border(exclusions, l, puzzle, (*standard_sudoku_singoverlap_regions(l), diagonal_sudoku_regions(l)), (), (exclude_thermo_rule_gen(thermo),), frozenset(range(1, l+1)))[1],
      (lambda x, border: print_border(x, exc_from_border(border, exclusions)), lambda x, border: print_candidate_border(x, exc_from_border(border, exclusions))))
def killer_cages_sudoku(puzzle, killer):
  l = len(puzzle)
  value_set = frozenset(range(1, l+1))
  #tuple(frozenset(x) for x in jigsaw_to_coords(killer_to_jigsaw(killer, l)).values()) #not mutual exclusive across value set
  #if killer cage is in a row, column or subsquare we dont need to include it as part of cell visibility since its redundant...  
  return (puzzle, standard_sudoku_mutex_regions(l), (killer_puzzle_gen([(x, y) for x, y in killer if x <= sum(value_set) and len(y) <= l]),), (exclude_killer_rule_gen(killer), exclude_cage_hidden_tuple_rule_gen([y for _, y in killer]), exclude_cage_mirror_rule_gen([y for _, y in killer])), value_set, None,
          (lambda x: print_border(x, exc_from_border(killer_to_jigsaw(killer, l), set())), lambda x: print_candidate_border(x, exc_from_border(killer_to_jigsaw(killer, l), set()))))
def get_mutex_cages(cages, mx=None): #n^2 instead of combinatorics 2^n
  multiples, pairdict = [{(i,) for i in range(len(cages))}, set()], {}
  for i, c in enumerate(cages):
    pairdict[i] = set()
    for j, cj in enumerate(cages[i+1:]):
      if len(c.intersection(cj)) == 0:
        multiples[1].add((i, i+1+j))
        pairdict[i].add(i+1+j)
  for l in range(3, len(cages) + 1 if mx is None else mx+1):
    multiples.append(set())
    for p in multiples[-2]:
      multiples[-1] |= set(tuple((*p, x)) for x in set.intersection(*(pairdict[z] for z in p)).difference(p))
  return multiples
def break_contained_cages(cages):
  addcages, removecages = [], []
  for i, cage in enumerate(cages):
    for c in cages[i+1:]:
      if c[1].issubset(cage[1]):
        removecages.append(cage)
        if len(c[1]) != len(cage[1]):
          remregion = (cage[0] - c[0], frozenset(cage[1]).difference(c[1]))
          if not remregion in cages and not remregion in addcages: addcages.append(remregion)
      elif cage[1].issubset(c[1]):
        removecages.append(c)
        remregion = (c[0] - cage[0], frozenset(c[1]).difference(cage[1]))
        if not remregion in cages and not remregion in addcages: addcages.append(remregion)
  return [c for c in cages if not c in removecages], addcages
def best_killer_cages(killer, mutex_rules, cell_visibility_rules, value_set):
  print(killer)
  l = len(value_set)
  tot = sum(value_set)
  cf = combination_freedom_gen(cell_visibility_rules, value_set, l)
  genkiller, combinedkiller = [(x, frozenset(y)) for x, y in killer], []
  gencf = {x: cf(x)[0] for x in genkiller}
  while True:
    while True:
      nextkiller = []
      for p in get_mutex_cages([y for _, y in genkiller]):
        for jp in p:
          j = [genkiller[j] for j in jp]
          combined = frozenset.union(*(q[1] for q in j))
          #if len(combined) != sum(len(k[1]) for k in j): continue #no overlaps
          #if any(len(r.intersection(combined)) == 0 for r in region): continue
          #mutexregions = [r for rg in mutex_rules for r in rg if len(r.intersection(combined)) != 0] #only rows/rows (always), rows/houses (sometimes), columns/columns (always), columns/houses (sometimes) are mutually exclusive
          #if not any(r.issubset(combined) for rg in mutex_rules for r in rg): #no extras
          cfgroup = sum(gencf[x] for x in j)  
          for regions in mutex_rules:
            region = [r for r in regions if len(r.intersection(combined)) != 0]
            allregion = frozenset.union(*region)
            if len(combined) != len(allregion): #combined.issubset(allregion) and 
              newreg = allregion.difference(combined)
              newregion = (tot * len(region) - sum(x for x, _ in j), newreg)
              if not any(r.issubset(newreg) for rg in mutex_rules for r in rg) and not any(r.issubset(newreg) for _, r in genkiller): #no subsuming or repeated regions
                newcf = cf(newregion)[0]
                if newcf <= cfgroup and not newregion in genkiller and not newregion in nextkiller:
                  print(newregion, newcf)
                  nextkiller.append(newregion); gencf[newregion] = newcf
      if len(nextkiller) == 0: break
      genkiller.extend(nextkiller)
      genkiller, nextkiller = break_contained_cages(genkiller)
      for g in nextkiller: genkiller.append(g); gencf[g] = cf(g)[0]
    nextkiller = []
    for p in get_mutex_cages([y for _, y in genkiller], 2)[1:]: #will assume for now that combinations of 2 is enough but probably need to compare combination freedom as less or equal not less - then also keep doing this in a loop
      for jp in p:
        j = [genkiller[j] for j in jp]
        #if any cage is not visible from the other cages, then should skip...
        combined = frozenset.union(*(q[1] for q in j))
        if not any(r.issubset(combined) for rg in mutex_rules for r in rg): #no extras
          cfgroup = sum(gencf[x] for x in j)
          newregion = (sum(x for x, _ in j), combined)
          newcf = cf(newregion)[0]
          if newcf < cfgroup and not newregion in genkiller and not newregion in combinedkiller:
            for g in cage_breakdown(cell_visibility_rules, value_set, l, newregion):
              if not g in genkiller and not g in nextkiller: nextkiller.append(g); gencf[g] = cf(g)[0]
            combinedkiller.append(newregion); gencf[newregion] = newcf
            print(newregion)
            #for x in j:
            #  if not x in killer: genkiller.remove(x)
    if len(nextkiller) == 0: break
    genkiller.extend(nextkiller)
    genkiller, nextkiller = break_contained_cages(genkiller)
    for g in nextkiller: genkiller.append(g); gencf[g] = cf(g)[0]
    print(genkiller)
  return [(x, tuple(y)) for x, y in genkiller + combinedkiller]
def knight_killer_balanced_cages_sudoku(puzzle, killer):
  l = len(puzzle)
  value_set = frozenset(range(1, l+1))
  tot = sum(value_set)
  #totkiller = sum(x for x, _ in killer)
  #killerpoints = set.union(*(set(x) for _, x in killer))
  mutex_rules = standard_sudoku_mutex_regions(l)
  addkiller = [(x, set(y)) for x, y in killer if len(y) > l]
  for x, y in addkiller:
    regions = [r for rg in mutex_rules for r in rg if len(r.intersection(y)) != 0]
    reg1, reg2 = None, None
    for p in get_mutex_cages(regions):
      for jp in p:
        j = [regions[j] for j in jp]
        region = frozenset.union(*j)
        if not region.issuperset(y): continue
        if len(jp) == 2: reg1 = region
        if len(jp) == 3: reg2 = j
    if not reg1 is None and not reg2 is None:
      if tuple(len(reg2[i].intersection(y)) for i in range(3)) in ((3,3,6),(3,6,3),(6,3,3)):
        remcells = reg1.difference(y)
        addkiller.append((tot * (len(reg1) // l) - x, remcells))
        for i in range(3):
          if len(remcells.intersection(reg2[i])) == 3:
            addkiller.append(((x << 1) - (tot * (len(reg1) // l)) >> 1, reg2[i].intersection(y)))
  while True:
    addkiller, morekiller = break_contained_cages(addkiller)
    addkiller = addkiller + morekiller
    if len(morekiller) == 0: break
  print(addkiller)
  """
  for x, y in killer:
    for regions in mutex_rules:
      for region in regions:
        if len(region.intersection(y)) == len(y):
          killer = ((tot - x, tuple(region.difference(y))), *killer)
  for regions in mutex_rules:
    for region in regions:
      intsct = [(x, y) for x, y in killer if len(region.intersection(y)) != 0]
      for p in range(2, len(intsct)+1):
        for j in itertools.combinations(intsct, p):
          combined = frozenset.union(*(frozenset(q[1]) for q in j))
          if len(combined) != sum(len(k[1]) for k in j): continue #no overlaps
          #if len(region.intersection(combined)) != len(combined) and len(region.intersection(combined)) > (len(value_set) >> 1) and len(combined.difference(region)) <= (len(value_set) >> 1):
          #  newregion = (sum(q[0] for q in j), tuple(combined))
          #  if not newregion in addkiller:
          #    print(newregion)
          #    addkiller.append(newregion)
          for rs in mutex_rules:
            for r in itertools.combinations(rs, 2):
              fullr = frozenset.union(*r)
              if len(fullr.intersection(combined)) == len(combined) and all(len(z.intersection(combined)) != 0 for z in r) and len(fullr) != len(combined):
                newregion = (tot * len(r) - sum(x for x, _ in j), tuple(fullr - combined))
                if not newregion in killergen and not newregion in killer and newregion[0] <= tot: killergen.append(newregion)
          if len(region.intersection(combined)) == len(region):
            extras = [r for rg in mutex_rules for r in rg if len(r.intersection(combined)) == len(r)]
            if len(frozenset.union(*extras)) in [len(extras) * l, (len(extras) * l) >> 1]:
              if len(combined - frozenset.union(*extras)) != 0 and len(combined - frozenset.union(*extras)) <= l + (l >> 1):
                newregion = (sum(x for x, _ in j) - tot * (len(frozenset.union(*extras)) // l), tuple(combined - frozenset.union(*extras)))
                if not newregion in killergen:
                  killergen.append(newregion)
  cell_visibility_rules = mutex_regions_to_visibility(mutex_rules)
  deletekiller = []
  for x, y in killer:
    regions = [r for rg in mutex_rules for r in rg if len(r.intersection(y)) != 0]
    for j in itertools.combinations(regions, 2):      
      if len(frozenset.union(*j)) == len(j) * l: #non overlapping - len(j[0].intersection(j[1])) == 0
        exc = frozenset.union(*j).difference(y)
        if len(exc) != 0 and len(exc) <= l and len(frozenset.union(*j).intersection(y)) == len(y):
          killergen.append(((tot << 1) - x, tuple(exc)))
          #these are the balancing rule - should just look at regions directly this is actually a lousy way to find them
          addkiller.append((x - tot, tuple(z for z in j[0].intersection(y) if len(cell_visibility(z[0], z[1], l, cell_visibility_rules).intersection(j[1].intersection(y))) == 0)))
          addkiller.append((x - tot, tuple(z for z in j[1].intersection(y) if len(cell_visibility(z[0], z[1], l, cell_visibility_rules).intersection(j[0].intersection(y))) == 0)))
          deletekiller.append((x, y))
  addkiller.append((sum(value_set) * l - totkiller, tuple(set(itertools.product(list(range(l)), list(range(l)))) - killerpoints)))
  """
  cages = [(x, y) for x, y in killer if len(y) <= l] #only non-balanced cages
  cvr = (knight_rule_points,killer_puzzle_gen(cages),)
  cell_visibility_rules = mutex_regions_to_visibility(mutex_rules) + cvr
  killergen = best_killer_cages(cages + addkiller, mutex_rules, cell_visibility_rules, value_set) #+ [(3, ((6,4),))]
  #print(tuple(combination_freedom_gen(cell_visibility_rules, value_set, l)(x) for x in killergen))
  killergen = tuple(sorted(killergen, key=combination_freedom_gen(cell_visibility_rules, value_set, l)))
  #tuple(frozenset(x) for x in jigsaw_to_coords(killer_to_jigsaw(killer, l)).values()) #not mutual exclusive across value set
  #if killer cage is in a row, column or subsquare we dont need to include it as part of cell visibility since its redundant...  
  #print(cages, killergen)
  #exclude_cage_mirror_rule_gen([y for _, y in cages])
  return (puzzle, mutex_rules, cvr, (exclude_killer_rule_gen(killergen), exclude_cage_hidden_tuple_rule_gen([y for _, y in cages]), ), value_set, None,
          (lambda x: print_border(x, exc_from_border(killer_to_jigsaw(killer, l), set())), lambda x: print_candidate_border(x, exc_from_border(killer_to_jigsaw(killer, l), set()))))
def killer_cages_diagonal_sudoku(puzzle, killer):
  l = len(puzzle)
  #tuple(frozenset(x) for x in jigsaw_to_coords(killer_to_jigsaw(killer, l)).values()) #not mutual exclusive across value set
  #if killer cage is in a row, column or subsquare we dont need to include it as part of cell visibility since its redundant...
  return (puzzle, (*standard_sudoku_mutex_regions(l), diagonal_sudoku_regions(l)), (killer_puzzle_gen(killer),), (exclude_killer_rule_gen(killer), exclude_cage_hidden_tuple_rule_gen([y for _, y in killer]), exclude_cage_mirror_rule_gen([y for _, y in killer])), frozenset(range(1, l+1)), None,
          (lambda x: print_border(x, exc_from_border(killer_to_jigsaw(killer, l), set())), lambda x: print_candidate_border(x, exc_from_border(killer_to_jigsaw(killer, l), set()))))
def renban_sudoku(puzzle, cages):
  l = len(puzzle) #exclude_cage_mirror_rule_gen(cages)
  return (puzzle, standard_sudoku_mutex_regions(l), (jigsaw_points_gen(cages),), (exclude_renban_rule_gen(cages), exclude_cage_hidden_tuple_rule_gen(cages),), frozenset(range(1, l+1)), None,
          (lambda x: print_border(x, exc_from_border(cages_to_jigsaw(cages, l), set())), lambda x: print_candidate_border(x, exc_from_border(cages_to_jigsaw(cages, l), set()))))
def renban_killer_clue_combos(clue, cagesize, value_set, renban):
  odds, evens = {1, 3, 5, 7, 9}, [{2, 4, 6, 8}, {0, 2, 4, 6, 8}]
  mins, maxes = get_min_max_sums(value_set)
  svals = list(sorted(value_set))
  if renban: contigs = [sum(svals[i:i+cagesize]) for i in range(len(svals) - cagesize + 1)]
  cluenums = [{x * (10 ** i) for x in odds} if c == 'O' else ({x * (10 ** i) for x in evens[i == 0]} if c == 'E' else {x * (10 ** i) for x in odds.union(evens[i == 0])}) for i, c in enumerate(reversed(clue))]
  cluenums = [x+y for x, y in itertools.product(*cluenums)] if len(cluenums) != 1 else cluenums[0]
  return (renban, tuple(sorted(filter(lambda x: x >= mins[cagesize-1] and x <= maxes[cagesize-1] and (not renban or x in contigs), cluenums))))
def renban_killer_sudoku(puzzle, cages, evens):
  l = len(puzzle) #exclude_cage_mirror_rule_gen(cages)
  value_set = frozenset(range(1, l+1))
  killer = [(renban_killer_clue_combos(x, len(y), value_set, renban), y) for renban, x, y in cages if x != '']
  renbans = [[(renban, y)] for renban, x, y in cages if x == '']
  jigcages = [y for _, _, y in cages]
  mutex_rules = standard_sudoku_mutex_regions(l)
  combined_renban = []
  for regions in mutex_rules:
    for region in regions:
      for comb in itertools.combinations(cages, 2):
        combined = frozenset.union(*(frozenset(y) for _, _, y in comb))
        if len(region.intersection(combined)) == len(combined):
          renbans.append([(renban, y) for renban, _, y in comb])
  #excluded = frozenset(itertools.product(range(l), range(l))) - frozenset.union(*(frozenset(y) for _, y in cages))
  return (puzzle, standard_sudoku_mutex_regions(l), (killer_puzzle_gen([(x, y) for _, x, y in cages]),), (exclude_even_odd_gen(evens, True), exclude_renban_rule_gen(renbans), exclude_killer_rule_gen(killer), exclude_cage_hidden_tuple_rule_gen(jigcages),), value_set, None,
          (lambda x: print_border(x, exc_from_border(cages_to_jigsaw(jigcages, l), set())), lambda x: print_candidate_border(x, exc_from_border(cages_to_jigsaw(jigcages, l), set()))))  
def in_order_sudoku(puzzle, clues):
  l = len(puzzle)
  c = [(x, row_points(i, 0, l)) for i, x in enumerate(clues[0]) if not x is None] + [(x, column_points(0, i, l)) for i, x in enumerate(clues[1]) if not x is None]
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_in_order_rule_gen(c),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def get_diagonal(side, index, l):
  if side == 0: #left side diagonals heading down and right
    return tuple((i + index, i) for i in range(l - index))
  elif side == 1: #right side diagonals heading up and left, excluding last shared diagonal with left
    return tuple((i, i + index) for i in range(l - index))
  elif side == 2: #top side diagonals heading down and left
    return tuple((i, index - i) for i in range(index + 1))
  elif side == 3: #bottom diagonls heading up and right, excluding first shared diagonal with top
    return tuple((l - i - 1, index + i + 1) for i in range(l - index - 1))
def cage_breakdown(cell_visibility_rules, value_set, l, k):
  groups = get_mutex_cells(k[1], cell_visibility_rules, l)
  groupcand = get_mutex_groups_min_dependent(groups)
  sums = multi_cage_sums(k[0], [[f for j, f in g] for g in groupcand], value_set)
  newcage = []
  for i, x in enumerate(sums):
    if len(x) == 1:
      newcage.append((next(iter(x)), frozenset(groups[i])))
  return newcage
def combination_freedom_gen(cell_visibility_rules, value_set, l):
  mins, maxes = get_min_max_sums(value_set)
  def combination_freedom(k):
    groups = get_mutex_cells(k[1], cell_visibility_rules, l)
    groupcand = get_mutex_groups_min_dependent(groups)
    return (num_multi_cage_combs(k[0], [len(g) for g in groupcand if g != []], value_set), len(k[1]), k[0])
    """
    covered = set()
    mn, mx = 0, 0
    for g in get_mutex_cells(k[1], cell_visibility_rules, l):
      ln = len(g - covered)
      if ln == 0: continue
      mn += mins[ln-1]
      mx += maxes[ln-1]
      covered |= g      
    return min(k[0] - mn, mx - k[0])
    """
  return combination_freedom
def little_killer_sudoku(puzzle, little_killer_diagonals):
  l = len(puzzle)
  killer = [(y, get_diagonal(i, j, l)) for i, x in enumerate(little_killer_diagonals) for j, y in enumerate(x) if not y is None]
  mutex_rules = standard_sudoku_mutex_regions(l)
  value_set = frozenset(range(1, l+1))
  addkiller = []
  for regions in mutex_rules:
    for region in regions:
      intsct = [(x, y) for x, y in killer if len(region.intersection(y)) != 0]
      for p in range(2, len(intsct)+1):
        for j in itertools.combinations(intsct, p):
          combined = frozenset.union(*(frozenset(q[1]) for q in j))
          if len(region.intersection(combined)) > (len(value_set) >> 1) and len(combined.difference(region)) <= (len(value_set) >> 1):
            addkiller.append((sum(q[0] for q in j), tuple(combined)))
  #killer clues should be sorted by number of combinations for maximal speed very similar to a degrees of freedom metric  
  cvr = ()
  cell_visibility_rules = mutex_regions_to_visibility(mutex_rules) + cvr
  killer = list(sorted((*killer, *addkiller), key=combination_freedom_gen(cell_visibility_rules, value_set, l)))
  return (puzzle, mutex_rules, cvr, (exclude_killer_rule_gen(killer),), value_set, None, (print_sudoku, print_candidate_format))
def sandwich_sudoku(puzzle, sandwich_row_cols):
  l = len(puzzle)
  sandwiches = [(x, row_points(i, 0, l)) for i, x in enumerate(sandwich_row_cols[0]) if not x is None] + [(x, column_points(0, i, l)) for i, x in enumerate(sandwich_row_cols[1]) if not x is None]
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_sandwich_rule_gen(sandwiches),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def battlefield_sudoku(puzzle, battlefield_row_cols):
  l = len(puzzle)
  fields = [(x, row_points(i, 0, l)) for i, x in enumerate(battlefield_row_cols[0]) if not x is None] + [(x, column_points(0, i, l)) for i, x in enumerate(battlefield_row_cols[1]) if not x is None]
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_battlefield_rule_gen(fields),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def sandwich_arrow_sudoku(puzzle, arrows, increasing, sandwich_row_cols):
  l = len(puzzle)
  sandwiches = [(x, row_points(i, 0, l)) for i, x in enumerate(sandwich_row_cols[0]) if not x is None] + [(x, column_points(0, i, l)) for i, x in enumerate(sandwich_row_cols[1]) if not x is None]
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_arrow_rule_gen(arrows, increasing), exclude_sandwich_rule_gen(sandwiches),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format)) 
def knight_killer_cages_even_mirror_magic_square_sudoku(puzzle, killer_cages, magic_squares, evens, mirror_sub_squares):
  l = len(puzzle)
  ms = [magic_square_rows((x[0], x[1]), x[2], x[3]) for x in magic_squares]
  killers = [magic_square_center_to_killer_cages(x, frozenset(range(1, l+1))) for x in ms]
  mirrors = tuple(zip(*(sorted(get_sub_square_points(sub_square_from_point(m[0], m[1], l), l)) for x in mirror_sub_squares for m in x)))
  return (puzzle, standard_sudoku_mutex_regions(l), (knight_rule_points,mirror_visibility_gen(mirrors, standard_sudoku_mutex_regions(l), (knight_rule_points,))), (exclude_magic_square_rule_gen(ms), *(exclude_killer_rule_gen(killer) for killer in killers), exclude_killer_rule_gen(killer_cages), exclude_even_odd_gen(evens, True), exclude_mirror_gen(mirrors)), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def fibonacci_rule_sudoku(puzzle, cages):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_fibonacci_rule_gen(cages),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def manhattan_thermo_doubles_sudoku(puzzle, thermo, doubles, taxis, max_subset):
  l = len(puzzle)
  return (puzzle, (*standard_sudoku_mutex_regions(l), diagonal_sudoku_regions(l)), (), (exclude_thermo_rule_gen(thermo),exclude_double_rule_gen(doubles),exclude_manhattan_rule_gen(taxis,max_subset)), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def manhattan_sudoku(puzzle, taxis, max_subset):
  l = len(puzzle)
  return (puzzle, (*standard_sudoku_mutex_regions(l), diagonal_sudoku_regions(l)), (), (exclude_manhattan_rule_gen(taxis,max_subset),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))

def file_puzzles(filename):
  with open(filename) as f:
    return [standard_sudoku(str_to_sudoku(x.strip())) for x in f.readlines() if len(x) == 9 * 9 + 1]

def get_ctc_puzzles():
  pi_king_sudoku = ( #https://www.youtube.com/watch?v=N41yZsxIsK8
    (None, None, None, 4, 3, 1, None, None, None),
    (None, None, 8, None, None, None, 4, None, None),
    (None, 3, None, None, None, None, None, 1, None),
    (2, None, None, None, None, None, None, None, 5),
    (3, None, None, None, 6, None, None, None, 9),
    (9, None, None, None, None, None, None, None, 2),
    (None, 7, None, None, None, None, None, 6, None),
    (None, None, 9, None, None, None, 5, None, None),
    (None, None, None, 8, 5, 3, None, None, None))

  diff_sudoku = ( #https://www.youtube.com/watch?v=BnmxrrZUgbk
    (None, None, None, 1, None, 2, None, None, None),
    (None, 6, None, None, None, 8, 3, None, None),
    (5, None, None, None, None, None, None, None, 9),
    (None, None, None, 4, None, 7, None, None, 8),
    (6, 8, None, None, None, 5, None, None, None),
    (None, None, 4, None, None, None, None, 1, None),
    (None, 2, None, None, None, None, 5, None, None),
    (None, None, None, None, 7, None, 2, None, 6),
    (None, 9, None, None, None, 6, 7, None, None))
    
  nytimes_hard_sudoku = ( #https://www.youtube.com/watch?v=QgkVz9sdHEs
    (6, 3, None, None, None, None, None, 8, 1),
    (None, 2, None, None, None, 3, None, None, None),
    (None, None, None, None, 1, 7, 4, 3, None),
    (None, 9, 6, 4, None, None, 5, 7, None),
    (None, None, None, 7, 6, 2, None, None, None),
    (None, 8, None, None, None, None, 6, None, None),
    (None, 6, None, None, 2, None, None, None, None),
    (3, None, 9, None, None, None, None, 6, None),
    (None, None, None, None, None, None, None, None, 9))

  very_hard_sudoku = ( #https://www.youtube.com/watch?v=oPTe52OmMEk
    (None, 7, None, 2, 5, None, 4, None, None),
    (8, None, None, None, None, None, 9, None, 3),
    (None, None, None, None, None, 3, None, 7, None),
    (7, None, None, None, None, 4, None, 2, None),
    (1, None, None, None, None, None, None, None, 7),
    (None, 4, None, 5, None, None, None, None, 8),
    (None, 9, None, 6, None, None, None, None, None),
    (4, None, 1, None, None, None, None, None, 5),
    (None, None, 7, None, 8, 2, None, 3, None))
    
  technique_sudoku = ( #https://www.youtube.com/watch?v=pWqUzSgJCE0
    (8, None, None, None, None, 6, 3, None, 5),
    (None, 4, None, None, None, None, None, 7, None),
    (None, None, None, None, None, None, None, None, None),
    (None, 1, None, None, 3, 8, 7, None, 4),
    (None, None, None, 1, None, 4, None, None, None),
    (3, None, None, None, 7, None, 2, 9, None),
    (None, None, None, None, None, 3, None, None, None),
    (None, 2, None, None, None, None, None, 4, None),
    (5, None, 6, 8, None, None, None, None, 2))
    
  champion_sudoku = ( #https://www.youtube.com/watch?v=teOXgjSK7vA
    (None, 7, None, None, None, 4, None, None, 2),
    (None, None, 1, None, 3, None, None, 4, None),
    (None, None, None, 5, None, None, 1, None, None),
    (None, 4, None, None, None, 3, None, None, 8),
    (None, None, 3, None, None, None, 7, None, None),
    (1, None, None, 6, None, None, None, 9, None),
    (None, None, 4, None, None, 1, None, None, None),
    (None, 2, None, None, 7, None, 8, None, None),
    (5, None, None, 9, None, None, None, 6, None))
    
  diabolical_sudoku = ( #https://www.youtube.com/watch?v=eBs5N99m1aA
    (1, 2, None, 3, None, 4, None, 7, 8),
    (3, None, None, 8, None, 7, None, None, 9),
    (None, None, None, None, None, None, None, None, None),
    (2, 9, None, None, None, None, None, 8, 4),
    (None, None, None, None, 8, None, None, None, None),
    (7, 8, None, None, None, None, None, 1, 3),
    (None, None, None, None, None, None, None, None, None),
    (4, None, None, 9, None, 2, None, None, 6),
    (9, 3, None, 1, None, 8, None, 4, 7))
    
  swordfish_sudoku = ( #https://www.youtube.com/watch?v=7BX0FsQxpfQ
    (None, 7, None, None, 4, None, 2, None, None),
    (None, None, None, 3, None, None, None, 7, 9),
    (5, None, 6, None, 9, None, 4, None, None),
    (None, None, None, 4, None, None, None, 5, None),
    (None, None, 7, None, None, None, 3, None, None),
    (None, 3, None, None, None, 8, None, None, None),
    (None, None, 1, None, 6, None, 7, None, 3),
    (7, 6, None, None, None, 9, None, None, None),
    (None, None, 2, None, 1, None, None, 8, None))
    
  advanced_sudoku = ( #https://www.youtube.com/watch?v=fJYV7xddl-g
    (None, 4, 7, None, 2, None, 3, None, None),
    (None, None, 6, 9, None, None, 4, None, None),
    (2, None, None, None, None, 8, None, None, None),
    (5, None, None, None, None, 9, None, None, 1),
    (None, 9, None, None, 6, None, None, 8, None),
    (1, None, None, 7, None, None, None, None, 5),
    (None, None, None, 8, None, None, None, None, 2),
    (None, None, 8, None, None, 3, 1, None, None),
    (None, None, 1, None, 9, None, 8, 6, None))
    
  hardest_sudoku = ( #https://www.youtube.com/watch?v=-ZZFEgCQsvA
    (None, 8, 1, None, 6, None, None, 5, 9),
    (None, None, None, None, None, 3, None, 2, None),
    (None, 6, None, None, 8, None, None, None, None),
    (4, None, None, None, None, None, 5, None, None),
    (None, 2, None, None, None, None, None, None, None),
    (None, 7, None, 2, None, None, 4, 8, None),
    (8, None, None, None, None, None, 9, None, 5),
    (7, None, None, 6, None, 9, None, 3, None),
    (None, None, 5, None, None, None, None, 4, None))
    
  us_championship_knights_sudoku = ( #https://www.youtube.com/watch?v=tHXXCW15bsk
    (6, None, None, None, None, None, None, 8, 9),
    (None, None, None, None, None, None, None, None, None),
    (None, None, 1, 2, 3, None, None, None, None),
    (None, None, 4, 5, 6, None, None, None, None),
    (None, None, 7, 8, 9, None, None, None, None),
    (None, None, None, None, None, None, 4, None, None),
    (None, None, None, None, None, 2, None, None, None),
    (3, None, None, None, None, None, None, 1, 2),
    (7, None, None, None, None, None, None, 4, 5))
    
  expert_knights_sudoku = ( #https://www.youtube.com/watch?v=mTdhTfAhOI8
    (None, 3, None, None, 4, 1, None, None, 7),
    (None, None, None, 5, None, None, None, None, None),
    (None, None, None, 8, None, 9, None, None, None),
    (6, None, None, None, None, None, None, 7, None),
    (None, None, None, None, None, None, None, None, 4),
    (None, 4, None, None, None, None, None, None, None),
    (3, None, None, None, None, None, None, None, None),
    (None, None, None, None, 6, None, None, 5, None),
    (None, 6, 4, 3, None, None, None, None, None))
    
  extreme_sudoku = ( #https://www.youtube.com/watch?v=icVgYHj2_PA
    (None, 2, None, 1, 8, None, None, 3, None),
    (None, None, None, 3, None, 6, None, None, None),
    (None, None, 6, None, None, 4, None, None, None),
    (None, None, None, 5, None, None, 4, 1, None),
    (1, None, 5, None, None, None, 2, None, 8),
    (None, 9, 4, None, None, 2, None, None, None),
    (None, None, None, 2, None, None, 7, None, None),
    (None, None, None, 8, None, 5, None, None, None),
    (None, 6, None, None, 3, 1, None, 2, None))
    
  miracle_sudoku = ( #https://www.youtube.com/watch?v=yKf9aUIxdb4
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, 1, None, None, None, None, None, None),
    (None, None, None, None, None, None, 2, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None))
    
  prize_sudoku = ( #https://www.youtube.com/watch?v=v0Vz7g6EWTs
    (1, None, None, None, None, None),
    (None, 2, None, None, None, None),
    (None, None, 3, None, None, None),
    (None, None, None, 4, None, None),
    (None, None, None, None, 5, None),
    (None, None, None, None, 2, 6))
  prize_sudoku_jigsaw = (
    (4, 4, 4, 2, 2, 2),
    (4, 2, 2, 2, 3, 3),
    (4, 6, 6, 6, 6, 3),
    (4, 6, 5, 1, 1, 3),
    (5, 6, 5, 1, 3, 3),
    (5, 5, 5, 1, 1, 1))
  prize_sudoku_subsquare_exc = frozenset((
    ((0, 1), (1, 1)), ((1, 0), (1, 1)), ((1, 1), (2, 1)),
    ((2, 2), (3, 2)), ((2, 3), (3, 3)), ((3, 2), (3, 3)),
    ((3, 4), (4, 4)), ((4, 3), (4, 4)), ((4, 4), (5, 4)),
    ((0, 2), (0, 3)), ((5, 2), (5, 3))))
    
  nightmare_sudoku = ( #https://www.youtube.com/watch?v=Tv-48b-KuxI, https://www.patreon.com/posts/37223311
    (1, 2, 3, 4, None, None),
    (None, None, None, None, None, None),
    (None, None, None, None, None, None),
    (None, None, None, None, None, None),
    (None, None, None, None, None, None),
    (None, None, 6, 5, None, None))
  nightmare_sudoku_subsquare_exc = frozenset((
    ((0, 2), (0, 3)), ((1, 1), (2, 1)), ((2, 0), (3, 0)),
    ((2, 5), (3, 5)), ((3, 2), (3, 3)), ((3, 4), (4, 4)),
    ((5, 2), (5, 3))))
  nightmare_sudoku_thermo = (((2, 1), (2, 0)), ((3, 4), (3, 5)))
  
  killer_xxl_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=qQ-B8R3wnEM
  killer_xxl_sudoku_cages = (
    (44, ((0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 8))),
    (44, ((2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 6), (7, 7), (7, 8))),
    (41, ((1, 7), (2, 7), (3, 5), (3, 6), (3, 7), (4, 7), (5, 7), (6, 7))),
    (36, ((1, 0), (1, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0))),
    (36, ((7, 2), (7, 3), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7))),
    (21, ((2, 2), (2, 3), (3, 3))),
    (21, ((3, 1), (3, 2), (4, 1))),
    (18, ((6, 1), (7, 1), (8, 0), (8, 1))),
    (16, ((4, 2), (5, 1), (5, 2), (6, 2))),
    (14, ((4, 4), (4, 5), (5, 4))),
    (14, ((5, 6), (6, 4), (6, 5), (6, 6))),
    (11, ((1, 5), (2, 5))),
    (7, ((7, 4), (7, 5))))
    
  thermo_app_sudoku = ( #https://www.youtube.com/watch?v=va3xrk7YALo
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, 8),
    (5, None, None, None, None, None, None, None, None),
    (None, None, None, None, 2, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, 7, None, None, None, None),
    (None, None, None, None, None, None, None, None, 6),
    (6, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None))
  thermo_app_sudoku_thermometers = (
    ((1, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (1, 1)),
    ((2, 1), (3, 2), (3, 3), (4, 4), (5, 5), (5, 6), (6, 7)),
    ((7, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (7, 1))
  )
  
  slippery_sandwich_sudoku = ( #https://www.youtube.com/watch?v=e5v4_Z1b_mg
    (None, None, None, None, None, None, None, None, None),
    (None, 4, None, None, 2, None, None, 3, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, 2, None, None, None, None, None, 1, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, 6, None, None, 9, None, None, 5, None),
    (None, None, None, None, None, None, None, None, None))
    
  slippery_sandwich_row_cols = ((17, 17, 17, 17, 20, 22, 27, 17, 0), (17, 17, 17, 17, 20, 22, 27, 17, 0))
  
  aad_tribute_sudoku = ( #https://www.youtube.com/watch?v=LuFvoFkqZDA
    (2, None, None, None, None, None, None, None, 5),
    (None, None, None, None, None, None, None, None, None),
    (8, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, 2),
    (None, None, None, None, None, None, None, None, None),
    (5, None, None, None, None, None, None, None, 8))
    
  aad_tribute_thermo = (((1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1)),)
  aad_magic_squares = ((4, 4, 3, 0),)
  
  magical_miracle_sudoku = ( #https://www.youtube.com/watch?v=LuFvoFkqZDA
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, 1, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, 2, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, 4, None, None, None, None, None))
  magical_miracle_squares = ((4, 4, 3, 2),)
  
  fibo_series_two_sudoku = ( #https://www.youtube.com/watch?v=hXiwVVxEH9g
    (None, None, None, None, 1, None, None, None, None),
    (1, None, None, None, None, None, None, 3, None),
    (None, None, None, None, None, 2, None, None, 5),
    (None, None, None, None, None, None, None, None, None),
    (None, 2, 1, None, None, None, 8, None, None),
    (None, None, None, 1, 3, None, None, None, None),
    (None, None, None, None, None, None, 5, None, None),
    (None, None, None, None, None, 5, None, None, None),
    (3, 4, None, None, None, None, None, 8, 9))
  fibo_series_two_arrows = (
    ((0, 3), (0, 4), (0, 5)),
    ((1, 5), (1, 6), (1, 7)),
    ((2, 0), (2, 1), (2, 2)),
    ((2, 3), (2, 4), (2, 5)),
    ((3, 6), (3, 5), (3, 4)),
    ((5, 0), (4, 0), (3, 0)),
    ((5, 1), (5, 2), (5, 3)),
    ((6, 2), (7, 2), (8, 2)),
    ((7, 8), (7, 7), (7, 6)),
    ((8, 4), (8, 3), (8, 2)),
    ((8, 5), (7, 5), (6, 5)))
    
  excellence_elegance_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=Vq0ZAf-tvGg
  excellence_elegance_arrows = (
    ((3, 3), (2, 2), (1, 1)),
    ((3, 5), (2, 6), (1, 7), (0, 8)),
    ((5, 3), (6, 2), (7, 1), (8, 0)),
    ((5, 5), (6, 6), (7, 7), (8, 8)))
  excellence_elegance_sandwich_row_cols = ((20, None, None, 20, 33, None, 0, None, None), (None, None, None, 27, None, 13, None, None, 13))    
  
  fiendish_little_killer_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=9DOjZUGef7A
  
  fiendish_little_killer_diagonals = (
    (None, None, None, None, 26, 25, 21, 13, None), #left side diagonals heading down and right
    (None, None, 20, None, 22, None, 20, None), #right side diagonals heading up and left, excluding last shared diagonal with left
    (None, 13, 15, 31, None, None, None, None, None), #top side diagonals heading down and left
    (None, None, None, None, None, None, 15, None) #bottom diagonls heading up and right, excluding first shared diagonal with top
    )
    
  miracle_man_thermo_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=bYXstZ8obAw
  miracle_man_thermos = (
    ((0, 3), (1, 3), (1, 2)),
    ((1, 8), (2, 8), (3, 8), (4, 8)),
    ((2, 7), (1, 7), (0, 7)),
    ((3, 0), (2, 0), (1, 0), (0, 0)),
    ((3, 1), (2, 1), (1, 1), (0, 1)),
    ((3, 4), (2, 4), (1, 4), (0, 4)),
    ((3, 6), (3, 5)),
    ((5, 4), (6, 4), (7, 4), (8, 4)),
    ((5, 7), (6, 7), (7, 7), (8, 7)),
    ((5, 8), (6, 8), (7, 8), (8, 8)),
    ((6, 1), (7, 1), (8, 1)),
    ((6, 2), (6, 3)),
    ((6, 5), (7, 5), (7, 6)),
    ((7, 0), (6, 0), (5, 0), (4, 0)))
    
  more_magic_sudoku = ( #https://www.youtube.com/watch?v=qCAQxEbgUVw
    (None, 7, None, None, None, None, 2, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, 8, 2, None),
    (None, None, None, None, None, None, None, None, None),
    (None, 5, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None))
  more_magic_sudoku_killer_cages = ((10, ((8, 6), (8, 7))),)
  more_magic_sudoku_magic_squares = ((4, 4, 3, 0),)
  more_magic_sudoku_evens = ((0, 7), (1, 8), (6, 7))
  more_magic_sudoku_mirror_sub_squares = (((4, 1), (7, 4)),)
  
  minimalist_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=5O1W893jCjc
  minimalist_thermos = (
    ((0, 3), (1, 2), (2, 1), (3, 0)),
    ((6, 8), (5, 7), (4, 6), (3, 5), (2, 4), (1, 3), (0, 2)),
    ((8, 3), (7, 4), (6, 5), (5, 6), (4, 7), (3, 8)))
    
  classic_technique_sudoku = (
    (None, 2, 1, None, None, 4, 9, 8, None),
    (4, None, None, None, None, 2, None, 6, None),
    (None, None, 5, None, 3, None, 2, None, None),
    (None, None, None, None, None, 7, None, None, None),
    (None, 5, None, None, 1, None, None, 9, None),
    (None, None, None, 2, None, None, None, None, None),
    (None, None, 4, None, 2, None, 1, None, None),
    (None, 9, None, 8, None, None, None, None, 7),
    (None, 1, 6, 9, None, None, 8, 3, None))
    
  killer_xxl_impressive_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=gzZl1EK4bww

  killer_xxl_impressive_sudoku_cages = (
    (42, ((0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (2, 5))),
    (33, ((0, 4), (0, 5), (0, 6), (1, 6), (1, 7), (2, 6), (2, 7))),
    (39, ((1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (4, 0))),
    (38, ((1, 8), (2, 8), (3, 6), (3, 7), (3, 8), (4, 6), (5, 6))),
    (32, ((3, 2), (4, 2), (5, 0), (5, 1), (5, 2), (6, 0), (7, 0))),
    (31, ((4, 8), (5, 8), (6, 6), (6, 7), (6, 8), (7, 6), (7, 7))),
    (37, ((6, 1), (6, 2), (7, 1), (7, 2), (8, 2), (8, 3), (8, 4))),
    (28, ((6, 3), (6, 4), (6, 5), (7, 5), (8, 5), (8, 6), (8, 7))))
  
  face_channel_thermo_sudoku = ( #https://www.youtube.com/watch?v=gtsPNINIioM
    (None, 8, 7, 6, None, 9, 3, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, 5),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
  )
  
  face_channel_thermo_sudoku_thermos = (
    ((2, 3), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)),
    ((2, 5), (1, 5), (1, 6), (1, 7), (2, 7), (1, 8)),
    ((5, 4), (4, 4), (3, 4), (4, 3), (5, 3)),
    ((3, 7), (3, 6), (3, 5)),
    ((6, 1), (7, 2), (7, 3), (7, 4)),
    ((6, 2), (6, 3)),
    ((6, 6), (6, 5)),
    ((6, 7), (7, 6)),
    ((7, 1), (8, 1), (8, 0), (7, 0), (6, 0)),
    ((7, 8), (8, 8), (7, 7), (8, 6), (8, 7)),
    ((8, 5), (7, 5), (6, 4))
  )
  
  battlefield_puzzle = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=YoKizMhyjGU
  battlefield_sudoku_overgaps = (
    (12, None, 21, 39, 39, 3, 5, None, 5),
    (2, None, 28, 6, 16, 12, 25, None, 9)
  )
  
  renban_puzzle = (
    (None, 2, 3, None, None, None, 9, None, None),
    (1, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, 1, None, None, None),
    (None, None, None, None, None, 8, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, 8, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, 7, None, None))
  renban_puzzle_contigs = (
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0)),
    ((0, 7), (0, 8), (1, 8), (2, 8), (3, 8)),
    ((1, 3), (1, 4), (1, 5), (1, 6), (2, 3)),
    ((2, 1), (3, 1), (4, 1), (5, 1), (5, 2)),
    ((3, 6), (3, 7), (4, 7), (5, 7), (6, 7)),
    ((5, 0), (6, 0), (7, 0), (8, 0), (8, 1)),
    ((6, 5), (7, 2), (7, 3), (7, 4), (7, 5)),
    ((7, 8), (8, 5), (8, 6), (8, 7), (8, 8)))
    
  clever_type_sudoku = ( #https://www.youtube.com/watch?v=A2QSKgkrQJs
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, 7, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None)
  )
  
  clever_type_sudoku_values = (
    ((4, 2, 3), (2, 7, 3), (1, 3, 7), (6, 2, 5), (2, 1, 6), (8, 1, 5), (1, 6, 2), (7, 4, 2), (3, 2, 4)),
    ((3, 4, 2), (1, 6, 4), (4, 2, 3), (4, 3, 2), (1, 4, 3), (4, 2, 3), (4, 3, 2), (2, 8, 5), (5, 4, 3)))
    
  king_knight_queen_sudoku = ( #https://www.youtube.com/watch?v=h9wQ73F8Cqc, https://www.youtube.com/watch?v=3HC6bb4s34M
    (None, None, None, None, None, None, None, None, None),
    (None, 6, None, None, None, None, None, None, 4),
    (3, 5, 1, 4, None, None, None, None, None),
    (None, None, None, None, 8, 6, 5, 9, 2),
    (6, None, None, None, None, None, 7, 4, 1),
    (None, None, 2, None, None, None, 8, None, None),
    (None, 7, None, None, None, None, None, None, None),
    (8, None, None, None, 2, None, 4, None, 7),
    (None, None, None, None, None, None, None, 5, 8))
    
  jousting_knights_puzzle = ( #https://www.youtube.com/watch?v=KDxxgJAv23A
    (1, None, None, None, 2, None, None, None, 3),
    (2, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, 4, None, None, None, None),
    (4, None, 7, None, 5, None, 3, None, 6),
    (None, None, None, None, 6, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, 8),
    (7, None, None, None, 8, None, None, None, 9)
  )
  jousting_knights_exclusions = ((0, 0), (0, 1), (0, 4), (0, 8), (1, 8), (3, 5), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8), (5, 3), (7, 0), (8, 0), (8, 4), (8, 7), (8, 8))
  
  best_ever_solve_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=vgKt7Q0HdzI
  best_ever_solve_sudoku_cages = (
    (40, ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5))),
    (68, ((0, 6), (1, 6), (2, 6), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (6, 6), (6, 7), (6, 8), (5, 8), (4, 8), (3, 8))),
    (53, ((0, 7), (0, 8), (1, 7), (1, 8), (2, 7), (2, 8), (3, 6), (3, 7), (4, 6), (4, 7), (5, 6), (5, 7))),
    (39, ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4))),
    (72, ((8, 2), (7, 2), (6, 2), (6, 3), (5, 3), (4, 3), (3, 3), (2, 3), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0))),
    (67, ((3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (6, 0), (6, 1), (7, 0), (7, 1), (8, 0), (8, 1))),
    (31, ((5, 4), (6, 4), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8))),
    (30, ((7, 3), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8))))
  
  tight_logic_mean_killer_sudoku = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=YsoETiwt84Q
  tight_logic_mean_killer_sudoku_cages = (
    (True, '', ((0, 0), (0, 1), (0, 2), (0, 3))),
    (True, '', ((0, 4), (0, 5), (0, 6), (1, 4))),
    (True, '?', ((0, 7), (0, 8))),
    (True, 'O?', ((1, 0), (2, 0))),
    (True, '?', ((1, 1), (1, 2))),
    (False, 'EE', ((1, 3), (2, 3), (2, 4), (3, 3), (3, 4))),
    (True, '', ((1, 5), (1, 6), (1, 7), (2, 5))),
    (True, '', ((1, 8), (2, 7), (2, 8), (3, 7))),
    (False, 'OO', ((2, 1), (2, 2))),
    (True, 'OE', ((2, 6), (3, 5), (3, 6), (4, 4), (4, 5))),
    (True, '?', ((3, 0), (3, 1), (3, 2))),
    (True, '?', ((3, 8), (4, 7), (4, 8))),
    (False, '', ((4, 0), (4, 1), (4, 2))),
    (True, '', ((4, 3), (5, 3))),
    (False, 'EO', ((4, 6), (5, 6), (5, 7), (5, 8))),
    (True, '?', ((5, 0), (6, 0), (7, 0))),
    (True, '', ((5, 1), (5, 2))),
    (False, 'O', ((5, 4), (5, 5))),
    (False, 'E', ((6, 1), (7, 1))),
    (False, 'OO', ((6, 2), (7, 2))),
    (True, '', ((6, 3), (6, 4), (7, 3))),
    (True, '', ((6, 5), (7, 4), (7, 5), (7, 6), (8, 5))),
    (False, '?', ((6, 6), (6, 7))),
    (True, 'E?', ((6, 8), (7, 7), (7, 8))),
    (True, '', ((8, 0), (8, 1))),
    (False, '', ((8, 2), (8, 3), (8, 4))),
    (True, 'O?', ((8, 6), (8, 7), (8, 8))))
  
  fibonacci_rule_puzzle = ( #https://www.youtube.com/watch?v=802_r7rBYms
    (None, None, None, None, None, None, None, None, None),
    (1, None, None, None, 2, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, 3, None, None, None, None, None),
    (None, None, None, None, None, 5, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, 8, None),
    (None, None, None, None, None, None, None, None, None),
    (4, None, None, None, None, None, 1, 3, None))
  fibonacci_rule_puzzle_cages = (
    (True, ((2, 2), (1, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 7), (6, 6), (7, 5), (8, 4), (7, 3), (6, 2), (5, 1), (4, 0))),
    (False, ((3, 2), (4, 3), (5, 4), (6, 3), (7, 2), (6, 1), (5, 0), (4, 1))))
  
  thermo_sandwich_puzzle = ( #https://www.youtube.com/watch?v=hSo8KlQoovU
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, 9, None, None, None, 1, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None))
    
  thermo_sandwich_puzzle_sandwiches = ((None, None, 15, None, 7, None, None, None, 7), (9, None, 3, None, 20, None, 11, None, None))
  thermo_sandwich_puzzle_thermos = (
    ((0, 4), (1, 4), (1, 5), (1, 6), (2, 6), (3, 6)),
    ((3, 4), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (3, 8)),
    ((5, 1), (4, 1), (3, 1)),
    ((6, 7), (7, 7), (8, 7)))
    
  technique_knight_sudoku = (
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, 7, None, 2),
    (None, None, 3, None, None, None, 6, None, None),
    (None, None, None, 5, None, None, None, None, None),
    (None, None, 1, 6, None, None, 3, None, None),
    (None, 5, 6, 4, None, None, None, None, None),
    (None, None, None, None, 1, None, None, 9, None),
    (None, None, None, None, 2, None, None, 7, None),
    (None, None, None, None, 3, None, None, None, 4))
    
  antiknight_killer_sudoku = ( #https://www.youtube.com/watch?v=Zk4qNEDXFSw
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, 1, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None))
  
  antiknight_killer_sudoku_cages = (
    (20, ((1, 1), (1, 2), (2, 1), (2, 2))),
    (15, ((1, 3), (1, 4), (2, 3), (2, 4))),
    (15, ((3, 1), (3, 2), (4, 1), (4, 2))),
    (10, ((3, 3), (3, 4), (4, 3), (4, 4))),
    (18, ((5, 3), (5, 4), (6, 4))),
    (12, ((5, 6), (5, 7), (5, 8))),
    (12, ((6, 5), (7, 5), (8, 5))),
    (7, ((6, 8), (7, 8))))
    
  manhattan_puzzle = ( #https://www.youtube.com/watch?v=ajvkwAqhGys
    (None, 4, None, None, None, None, None, None, None),
    (2, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, None, 1),
    (None, None, None, None, None, None, None, None, None),
    (None, None, None, None, None, None, None, 5, None),
    (None, None, 8, None, None, None, None, None, None))  
  manhattan_puzzle_thermos = (((0, 2), (0, 3)),)
  manhattan_puzzle_double = (((4, 3), (5, 3)),)
  manhattan_puzzle_taxis = ((0, 4), (0, 7), (1, 1), (2, 0), (2, 5), (3, 8), (4, 0), (5, 4), (6, 2), (7, 3), (8, 6))
  
  linked_sudoku = ( #https://www.youtube.com/watch?v=fu7zJExacac
    (None, None, None, 4, None, None),
    (None, None, 3, None, None, None),
    (None, 2, None, None, None, 5),
    (1, None, None, None, 4, None),
    (None, None, None, 5, None, None),
    (None, None, 2, None, None, None))
  linked_sudoku_jigsaw = get_jigsaw_rects(get_rects(3, 2, 6), 6)
  linked_sudoku_pair = (
    (None, None, 5, None, None, None),
    (None, None, None, 4, None, None),
    (3, None, None, None, 1, None),
    (None, 4, None, None, None, 2),
    (None, None, 3, None, None, None),
    (None, None, None, 5, None, None))
  linked_sudoku_pair_jigsaw = get_jigsaw_rects(get_rects(2, 3, 6), 6)
  #linked_sudoku_sets = linked_sudoku_set((jigsaw_sudoku(linked_sudoku, linked_sudoku_jigsaw),
  #                                        jigsaw_sudoku(linked_sudoku_pair, linked_sudoku_pair_jigsaw)))
  
  deficit_sudoku = ( #https://www.youtube.com/watch?v=fu7zJExacac
    (None, None, None, None, None, None, None),
    (None, None, None, 4, None, None, 5),
    (None, None, 3, None, 6, None, None),
    (None, 1, None, None, None, 2, None),
    (None, None, 5, None, None, None, 4),
    (None, None, None, 6, None, 1, None),
    (None, 2, None, None, 5, None, None))
  deficit_sudoku_jigsaw = (
    (0, 1, 1, 1, 1, 1, 1),
    (2, 3, 3, 3, 4, 4, 4),
    (2, 3, 3, 3, 4, 4, 4),
    (2, 5, 5, 5, 6, 6, 6),
    (2, 5, 5, 5, 6, 6, 6),
    (2, 7, 7, 7, 8, 8, 8),
    (2, 7, 7, 7, 8, 8, 8))
    
  surplus_sudoku = ( #https://www.youtube.com/watch?v=fu7zJExacac
    (1, None, None, 7, None, None, None),
    (None, 2, None, None, 6, None, None),
    (None, None, 3, None, None, 1, None),
    (7, None, None, None, None, None, 5),
    (None, 6, None, None, 5, None, None),
    (None, None, 4, None, None, 6, None),
    (None, None, None, 1, None, None, 7))
  surplus_sudoku_jigsaw = (
    (1, 2, 2, 3, 3, 3, 3),
    (1, 1, 2, 2, 2, 3, 3),
    (1, 1, 1, 2, 2, 2, 3),
    (4, 1, 1, 5, 6, 6, 3),
    (4, 7, 7, 7, 6, 6, 6),
    (4, 4, 7, 7, 7, 6, 6),
    (4, 4, 4, 4, 7, 7, 6))
  
  even_odd_small_big_puzzle = ( #https://www.youtube.com/watch?v=fu7zJExacac
    (None, None, None, None, None, None, None, None),
    (None, None, None, 8, None, 7, None, None),
    (None, 1, None, None, None, None, None, None),
    (None, None, None, None, None, None, 6, None),
    (None, 2, None, None, None, None, None, None),
    (None, None, None, None, None, None, 5, None),
    (None, None, 3, None, 4, None, None, None),
    (None, None, None, None, None, None, None, None))
  even_odd_small_big_puzzle_jigsaw = get_jigsaw_rects(get_rects(4, 2, 8), 8)
  even_odd_small_big_puzzle_clues = (
    (None, None, None, 'E', 'S', 'B', 'E', None),
    (None, 'S', 'E', None, 'B', None, None, None),
    (None, 'O', 'S', None, None, None, None, None),
    (None, None, None, None, None, 'E', 'S', None))
    
  border_skirmish_puzzle = ((None,) * 9,) * 9 #https://www.youtube.com/watch?v=YoKizMhyjGU
  border_skirmish_sudoku_overgaps = (
    (2, 2, 0, 1, 2, 1, 1, 1, 8),
    (9, 0, 0, 7, 0, 0, 8, 7, 0)
  )    
    
  #print_border(((None,) * 6,) * 6, prize_sudoku_subsquare_exc)
  #print_border(get_border_count([[None] * 6 for _ in range(6)], prize_sudoku_subsquare_exc), prize_sudoku_subsquare_exc)
  #print_border(add_region([[None] * 6 for _ in range(6)], max_region(region_ominos((3, 3), [[None] * 6 for _ in range(6)], prize_sudoku_subsquare_exc))[0], 1), prize_sudoku_subsquare_exc)
  #print_border(brute_border(prize_sudoku_subsquare_exc, 6)[1], exc_from_border(brute_border(prize_sudoku_subsquare_exc, 6)[1], prize_sudoku_subsquare_exc))
  #print_candidate_border(brute_border(prize_sudoku_subsquare_exc, 6)[0], prize_sudoku_subsquare_exc)
  #print_border(brute_border(nightmare_sudoku_subsquare_exc, 6)[1], exc_from_border(brute_border(nightmare_sudoku_subsquare_exc, 6)[1], nightmare_sudoku_subsquare_exc))
  #print_candidate_border(brute_border(nightmare_sudoku_subsquare_exc, 6)[0], nightmare_sudoku_subsquare_exc)
  #return ()
  return (
    #little_killer_sudoku(fiendish_little_killer_sudoku, fiendish_little_killer_diagonals),
    #knight_killer_balanced_cages_sudoku(antiknight_killer_sudoku, antiknight_killer_sudoku_cages),
    knight_killer_balanced_cages_sudoku(best_ever_solve_sudoku, best_ever_solve_sudoku_cages),

    thermo_sandwich_sudoku(thermo_sandwich_puzzle, thermo_sandwich_puzzle_sandwiches, thermo_sandwich_puzzle_thermos),
    )
  return (
    standard_sudoku(diff_sudoku),
    standard_sudoku(nytimes_hard_sudoku),
    standard_sudoku(very_hard_sudoku),
    standard_sudoku(technique_sudoku),
    standard_sudoku(champion_sudoku),
    standard_sudoku(diabolical_sudoku),
    standard_sudoku(swordfish_sudoku),
    standard_sudoku(advanced_sudoku),
    standard_sudoku(hardest_sudoku),
    standard_sudoku(extreme_sudoku),
    standard_sudoku(classic_technique_sudoku),
    arrow_sudoku(fibo_series_two_sudoku, fibo_series_two_arrows, False),
    sandwich_arrow_sudoku(excellence_elegance_sudoku, excellence_elegance_arrows, True, excellence_elegance_sandwich_row_cols),
    sandwich_sudoku(slippery_sandwich_sudoku, slippery_sandwich_row_cols),
    battlefield_sudoku(battlefield_puzzle, battlefield_sudoku_overgaps),
    battlefield_sudoku(border_skirmish_puzzle, border_skirmish_sudoku_overgaps),
    knight_thermo_magic_square_sudoku(aad_tribute_sudoku, aad_tribute_thermo, aad_magic_squares),
    king_knight_magic_square_sudoku(magical_miracle_sudoku, magical_miracle_squares),
    king_knight_orthagonal_sudoku(miracle_sudoku),
    jousting_knights_sudoku(jousting_knights_puzzle, jousting_knights_exclusions, set((5,))),
    manhattan_sudoku(manhattan_puzzle, manhattan_puzzle_taxis, 15),
    manhattan_thermo_doubles_sudoku(manhattan_puzzle, manhattan_puzzle_thermos, manhattan_puzzle_double, manhattan_puzzle_taxis, 15),
    fibonacci_rule_sudoku(fibonacci_rule_puzzle, fibonacci_rule_puzzle_cages),
    thermo_sudoku(thermo_app_sudoku, thermo_app_sudoku_thermometers),
    thermo_sudoku(miracle_man_thermo_sudoku, miracle_man_thermos),
    thermo_sudoku(face_channel_thermo_sudoku, face_channel_thermo_sudoku_thermos),
    thermo_sudoku_symmetry(minimalist_sudoku, minimalist_thermos),
    renban_sudoku(renban_puzzle, renban_puzzle_contigs),
    renban_killer_sudoku(tight_logic_mean_killer_sudoku, tight_logic_mean_killer_sudoku_cages, ((8, 7),)),
    in_order_sudoku(clever_type_sudoku, clever_type_sudoku_values),
    king_sudoku(pi_king_sudoku),
    king_sudoku(king_knight_queen_sudoku),
    knight_sudoku(technique_knight_sudoku),
    knight_sudoku(king_knight_queen_sudoku),
    knight_sudoku(us_championship_knights_sudoku),
    knight_sudoku(expert_knights_sudoku),
    queen_sudoku(king_knight_queen_sudoku,{9}),
    deficit_jigsaw_sudoku(deficit_sudoku, deficit_sudoku_jigsaw),
    surplus_jigsaw_sudoku(surplus_sudoku, surplus_sudoku_jigsaw),
    even_odd_small_big_jigsaw_sudoku(even_odd_small_big_puzzle, even_odd_small_big_puzzle_jigsaw, even_odd_small_big_puzzle_clues),
    jigsaw_sudoku(prize_sudoku, prize_sudoku_jigsaw),
    partial_border_jigsaw_sudoku(prize_sudoku, prize_sudoku_subsquare_exc),
    partial_border_diagonal_thermo_sudoku(nightmare_sudoku, nightmare_sudoku_subsquare_exc, nightmare_sudoku_thermo),
    little_killer_sudoku(fiendish_little_killer_sudoku, fiendish_little_killer_diagonals),
    killer_cages_sudoku(killer_xxl_sudoku, killer_xxl_sudoku_cages),
    killer_cages_diagonal_sudoku(killer_xxl_impressive_sudoku, killer_xxl_impressive_sudoku_cages),
    knight_killer_balanced_cages_sudoku(antiknight_killer_sudoku, antiknight_killer_sudoku_cages),
    knight_killer_balanced_cages_sudoku(best_ever_solve_sudoku, best_ever_solve_sudoku_cages),
    knight_killer_cages_even_mirror_magic_square_sudoku(more_magic_sudoku, more_magic_sudoku_killer_cages, more_magic_sudoku_magic_squares, more_magic_sudoku_evens, more_magic_sudoku_mirror_sub_squares),
    )

def get_impossible_puzzles():
  #https://www.theguardian.com/media/2010/aug/23/worlds-hardest-sudoku-solution
  prior_world_hardest_sudoku = ( #https://www.theguardian.com/media/2010/aug/22/worlds-hardest-sudoku
    (None, None, 5, 3, None, None, None, None, None),
    (8, None, None, None, None, None, None, 2, None),
    (None, 7, None, None, 1, None, 5, None, None),
    (4, None, None, None, None, 5, 3, None, None),
    (None, 1, None, None, 7, None, None, None, 6),
    (None, None, 3, 2, None, None, None, 8, None),
    (None, 6, None, 5, None, None, None, None, 9),
    (None, None, 4, None, None, None, None, 3, None),
    (None, None, None, None, None, 9, 7, None, None))

  #https://www.telegraph.co.uk/news/science/science-news/9360022/Worlds-hardest-sudoku-the-answer.html
  world_hardest_sudoku = ( #https://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html
    (8, None, None, None, None, None, None, None, None),
    (None, None, 3, 6, None, None, None, None, None),
    (None, 7, None, None, 9, None, 2, None, None),
    (None, 5, None, None, None, 7, None, None, None),
    (None, None, None, None, 4, 5, 7, None, None),
    (None, None, None, 1, None, None, None, 3, None),
    (None, None, 1, None, None, None, None, 6, 8),
    (None, None, 8, 5, None, None, None, 1, None),
    (None, 9, None, None, None, None, 4, None, None))

  return (
    standard_sudoku(prior_world_hardest_sudoku),
    standard_sudoku(world_hardest_sudoku))

def star_battle_regions(l, battle):
  return (*jigsaw_to_coords(battle).values(), *(row_points(x, 0, l) for x in range(l)), *(column_points(0, x, l) for x in range(l)))
"""
def cell_star_battle_rule(rem, battle, points, grouppoints):
  groups = jigsaw_to_coords(battle)
  l = len(rem)
  for (p, q) in frozenset.union(grouppoints, *(cell_visibility(i, j, l, (king_rule_points,)) for i, j in points)):
    if 1 in rem[p][q]:
      rem[p][q].remove(1)
  return rem
"""
def naked_single_star_battle(rem, battle, stars, groups):
  l, possible = len(rem), []
  for i in range(l):
    for j in range(l):
      if len(rem[i][j]) == 1 and 1 in rem[i][j]:
        exclude = []
        for (p, q) in frozenset.union(*king_rule_points(i, j, l)):
          if 1 in rem[p][q]: exclude.append((p, q, 1))
        if len(exclude) != 0: possible.append((exclude, NAKED_SINGLE, ()))
  return possible
def naked_multiples_star_battle(rem, battle, stars, groups):
  l, possible = len(rem), []
  for points in star_battle_regions(l, battle):
    found = {z for z in points if len(rem[z[0]][z[1]]) == 1 and 1 in rem[z[0]][z[1]]}
    if len(found) == stars:
      exclude = []
      for p in points - found:
        if 1 in rem[p[0]][p[1]]: exclude.append((p[0], p[1], 1))
      if len(exclude) != 0: possible.append((exclude, NAKED_SINGLE, ()))
  return possible
def hidden_multiples_star_battle(rem, battle, stars, groups):
  l, possible = len(rem), []
  for points in star_battle_regions(l, battle):
    exclusive = {z for z in points if 1 in rem[z[0]][z[1]]}
    if len(exclusive) == stars:
      exclude = []
      for p in exclusive:
        if 0 in rem[p[0]][p[1]]: exclude.append((p[0], p[1], 0))
      if len(exclude) != 0: possible.append((exclude, HIDDEN_SINGLE, ()))
  return possible
"""
def check_star_battle_rule(rem, battle):
  l, count = len(rem), 0
  for points in star_battle_regions(l, battle):
    exclusive = [(z, len(rem[z[0]][z[1]])) for z in points if 1 in rem[z[0]][z[1]]]
    exclusive = list(filter(lambda x: x[1] == 1, exclusive)) if len(exclusive) != 2 else (exclusive if exclusive[0][1] != 1 or exclusive[1][1] != 1 else [])
    if len(exclusive) == 2:
      count, rem = count + 1, cell_star_battle_rule(rem, battle, (exclusive[0][0], exclusive[1][0]), frozenset(points) - frozenset((exclusive[0][0], exclusive[1][0])))
      if 0 in rem[exclusive[0][0][0]][exclusive[0][0][1]]: rem[exclusive[0][0][0]][exclusive[0][0][1]].remove(0)
      if 0 in rem[exclusive[1][0][0]][exclusive[1][0][1]]: rem[exclusive[1][0][0]][exclusive[1][0][1]].remove(0)
    elif len(exclusive) == 1:
      rem = cell_star_battle_rule(rem, battle, (exclusive[0][0],), frozenset())
      if 0 in rem[exclusive[0][0][0]][exclusive[0][0][1]]: rem[exclusive[0][0][0]][exclusive[0][0][1]].remove(0)
  if count != 0: rem = check_star_battle_rule(rem, battle)
  return rem
"""
def get_squares(points):
  regions = []
  for i, j in points:
    if (i+1, j) in points and (i, j+1) in points and (i+1, j+1) in points:
      regions.append(frozenset(((i, j), (i+1, j), (i, j+1), (i+1, j+1))))
  return regions
def get_trominos(points):
  regions = []
  for i, j in points:
    if (i+1, j) in points and (i+1, j+1) in points and not (i, j+1) in points: #down, right |_
      regions.append(frozenset(((i, j), (i+1, j), (i+1, j+1))))
    if (i, j+1) in points and (i+1, j+1) in points and not (i+1, j) in points: #right, down -|
      regions.append(frozenset(((i, j), (i, j+1), (i+1, j+1))))
    if (i+1, j) in points and (i+1, j-1) in points and not (i, j-1) in points: #down, left _|
      regions.append(frozenset(((i, j), (i+1, j), (i+1, j-1))))
    if (i, j-1) in points and (i+1, j-1) in points and not (i+1, j) in points: #left, down |-
      regions.append(frozenset(((i, j), (i, j-1), (i+1, j-1))))
  return regions
def get_dominos(points):
  regions = []
  for i, j in points:
    if (i+1, j) in points and not (i, j-1) in points and not (i, j+1) in points and not (i+1, j-1) in points and not (i+1, j+1) in points:
      regions.append(frozenset(((i, j), (i+1, j))))
    if (i, j+1) in points and not (i-1, j) in points and not (i+1, j) in points and not (i-1, j+1) in points and not (i+1, j+1) in points:
      regions.append(frozenset(((i, j), (i, j+1))))
  return regions
def get_diag_dominos(points):
  regions = []
  for i, j in points:
    if (i+1, j+1) in points and not (i, j+1) in points and not (i+1, j) in points:
      regions.append(frozenset(((i, j), (i+1, j+1))))
    if (i+1, j-1) in points and not (i, j-1) in points and not (i+1, j) in points:
      regions.append(frozenset(((i, j), (i+1, j-1))))   
  return regions
def get_square_regions(points):
  squares = get_squares(points)
  if len(points) & 3 != 0: return None
  s = len(points) >> 2
  for comb in itertools.combinations(squares, s):
    u = frozenset() if s == 0 else frozenset.union(*comb)
    if len(u) == s << 2:
      return comb
  return None
def all_square_regions(l):
  return [[frozenset(((i, j), (i+1, j), (i, j+1), (i+1, j+1))) for j in range(0, l, 2)] for i in range(0, l, 2)]
def get_mutex_regions(points, numstars):
  #need all solutions to equation 4s+3t+2d+p=len(points), s+t+d+p=numstars naive way is easiest, perhaps Diophantine equation can be solved more efficiently
  #perms = []
  #a domino excludes its 4 length-wise orthagonal squares, a diagonal domino excludes the alternate corners in its square region
  #a tromino excludes its inner point in its square region
  combs = []
  #if numstars * 4 < len(points): return frozenset()
  squares = get_squares(points)
  for s in range(min(numstars, len(squares)), -1, -1):
    for comb in itertools.combinations(squares, s):
      u = frozenset() if s == 0 else frozenset.union(*comb)
      if len(u) == s << 2:
        r = points - u
        trominos = get_trominos(r)
        for t in range(min(numstars - s, len(trominos)), -1, -1):
          for combtrom in itertools.combinations(trominos, t):
            ut = frozenset() if t == 0 else frozenset.union(*combtrom)
            if len(ut) == t * 3:
              rt = r - ut
              dominos = get_dominos(rt)
              for d in range(min(numstars - s - t, len(dominos)), -1, -1):
                for combdom in itertools.combinations(dominos, d):
                  ud = frozenset() if d == 0 else frozenset.union(*combdom)
                  if len(ud) == d << 1:
                    rd = rt - ud
                    diag_dominos = get_diag_dominos(rd)
                    for dg in range(min(numstars - s - t - d, len(diag_dominos)), -1, -1):
                      for combddom in itertools.combinations(diag_dominos, dg):
                        udd = frozenset() if dg == 0 else frozenset.union(*combddom)
                        if len(udd) == dg << 1:
                          if len(u) + len(ut) + len(ud) + len(udd) + numstars - s - t - d - dg == len(points):
                            rdd = rd - udd
                            if any(len(rd.intersection(king_rule_pts(pt[0], pt[1]))) != 0 for pt in rdd): continue
                            #print(len(comb), len(combtrom), len(combdom))
                            combs.append(comb + combtrom + combdom + combddom + tuple(frozenset((x,)) for x in rdd))
  return combs
def region_space(rem, battle, stars, groups): #internal elimination, for 3 or more stars, this would need to be more complicated a check
  possible, l = [], len(rem)
  #central squares in region that needs 2 stars and can see all others is eliminated
  for points in groups.values():
    numstars = sum((1 in rem[i][j] and len(rem[i][j]) == 1 for i, j in points))
    if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
    rempoints = frozenset(filter(lambda p: len(rem[p[0]][p[1]]) != 1, points))
    exclude = []
    for i, j in rempoints:
      altpoints = frozenset.union(*king_rule_points(i, j, l))
      if all((x, y) in altpoints for x, y in rempoints - frozenset(((i, j),))):
        exclude.append((i, j, 1))
    if len(exclude) != 0: possible.append((exclude, LOCKED_CANDIDATES, ()))
  return possible
def bounded_regions(rem, battle, stars, groups): #row/column elimination
  possible, l = [], len(rem)
  regions = tuple(tuple(x) for x in standard_sudoku_singoverlap_regions(l))
  for p in range(2, l-1):
    for i in range(l-p+1):
      #number of bounded regions equal to number of rows/columns
      for region in regions:
        regionpoints = frozenset.union(*(region[i+q] for q in range(p)))
        tot = frozenset(tuple(battle[x][y] for x, y in regionpoints))
        contained = tuple(x for x in tot if all(p in regionpoints for p in groups[x]))
        if len(contained) == p:
          exclude = []
          points = set.union(*(groups[x] for x in contained))
          for x, y in regionpoints - points:
            if 1 in rem[x][y]: exclude.append((x, y, 1))
          if len(exclude) != 0: possible.append((exclude, LOCKED_CANDIDATES, ()))
        elif len(contained) != 0: pass
  return possible
def locked_candidates_star_battle(rem, battle, stars, groups):
  possible, l = [], len(rem)
  #jigsaw_visibility = jigsaw_points_gen(groups.values())
  for i in range(l):
    for j in range(l):
      #if (i == 2 or i == 3) and j == 1 or i == 2 and j == 4:
      #  if 1 in rem[i][j]: possible.append((((i, j, 1),), LOCKED_CANDIDATES, ()))
      neighborpoints = frozenset.union(*king_rule_points(i, j, l))
      if len(rem[i][j]) == 2: #external elimination, equivalent of claiming locked candidates
        altregions = frozenset(tuple(battle[p[0]][p[1]] for p in neighborpoints)) - frozenset((battle[i][j],))
        exclude = []
        rowvis, colvis = row_col_points(i, j, l)
        for x in altregions:
          numstars = sum((1 in rem[z][y] and len(rem[z][y]) == 1 for z, y in groups[x]))
          if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
          pts = tuple(filter(lambda p: len(rem[p[0]][p[1]]) == 2, groups[x] - neighborpoints))
          if len(pts) == 1 or len(pts) == 2 and len(get_dominos(pts)) == 1 or len(pts) == 3 and len(get_trominos(pts)) == 1 or len(pts) == 4 and len(get_squares(pts)) == 1 or set(pts).issubset(rowvis) or set(pts).issubset(colvis):
            #print(i, j, x, tuple(len(rem[p[0]][p[1]]) for p in groups[x] - neighborpoints), groups[x] - neighborpoints)
            exclude.append((i, j, 1))
            break
        if len(exclude) != 0: possible.append((exclude, LOCKED_CANDIDATES, ()))
        exclude = []
        for vispoints in row_col_points(i, j, l): #(*row_col_points(i, j, l), *jigsaw_visibility(i, j, l)):
          numstars = sum((1 in rem[x][y] and len(rem[x][y]) == 1 for x, y in vispoints))
          altregions = frozenset(tuple(battle[p[0]][p[1]] for p in vispoints)) - frozenset((battle[i][j],))
          if numstars == 0:
            for x in altregions:
              numstars = sum((1 in rem[z][y] and len(rem[z][y]) == 1 for z, y in groups[x]))
              if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
              pts = tuple(filter(lambda p: len(rem[p[0]][p[1]]) == 2, groups[x] - vispoints - neighborpoints))
              if len(pts) == 0:
                exclude.append((i, j, 1))
                break
              #elif len(pts) == 1 or len(pts) == 2 and len(get_dominos(pts)) == 1 or len(pts) == 3 and len(get_trominos(pts)) == 1 or len(pts) == 4 and len(get_squares(pts)) == 1:
              #  for pt in frozenset.intersection(*(frozenset.union(*king_rule_points(i, j, l)) for i, j in vispoints - groups[x] - neighborpoints)):
              #    if 1 in rem[pt[0]][pt[1]]: exclude.append((pt[0], pt[1], 1))
          numstars = sum((1 in rem[x][y] and len(rem[x][y]) == 1 for x, y in vispoints))
          if numstars != 1: continue
          for x in altregions:
            numstars = sum((1 in rem[z][y] and len(rem[z][y]) == 1 for z, y in groups[x]))
            if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
            pts = tuple(filter(lambda p: len(rem[p[0]][p[1]]) == 2, groups[x] - vispoints - neighborpoints))
            if len(pts) == 1 or len(pts) == 2 and len(get_dominos(pts)) == 1 or len(pts) == 3 and len(get_trominos(pts)) == 1 or len(pts) == 4 and len(get_squares(pts)) == 1:
              #print(i, j, x, tuple(len(rem[p[0]][p[1]]) for p in groups[x] - vispoints), groups[x] - vispoints)
              exclude.append((i, j, 1))
              break
        if len(exclude) != 0: possible.append((exclude, LOCKED_CANDIDATES, ()))
      elif len(rem[i][j]) == 1 and 1 in rem[i][j]:
        exclude = []
        for vispoints in row_col_points(i, j, l):
          numstars = sum((1 in rem[x][y] and len(rem[x][y]) == 1 for x, y in vispoints))
          if numstars != 1: continue
          altregions = frozenset(tuple(battle[p[0]][p[1]] for p in vispoints)) - frozenset((battle[i][j],))
          for x in altregions:
            numstars = sum((1 in rem[z][y] and len(rem[z][y]) == 1 for z, y in groups[x]))
            if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
            pts = tuple(filter(lambda p: len(rem[p[0]][p[1]]) == 2, groups[x] - vispoints - neighborpoints))
            if len(pts) == 1:
              #print(i, j, pts, x, tuple(len(rem[p[0]][p[1]]) for p in groups[x] - vispoints), groups[x] - vispoints)
              exclude.append((pts[0][0], pts[0][1], 0))
              break
        if len(exclude) != 0: possible.append((exclude, LOCKED_CANDIDATES, ()))
  return possible
def square_reduction(rem, battle, stars, groups):
  possible, l = [], len(rem)
  #solve by looking at a reduction into squares of the whole grid
  #totstars = l * stars
  if l & 1 == 0:
    allsquares = all_square_regions(l)
    squaremap = [[set((1,)) if any(len(rem[p[0]][p[1]]) == 1 and 1 in rem[p[0]][p[1]] for p in y) else (set((0,)) if all(len(rem[p[0]][p[1]]) == 1 and 0 in rem[p[0]][p[1]] for p in y) else set((0, 1))) for y in x] for x in allsquares] #if any star, then its a star, if all cannot be a star then no star, otherwise dual values
    def get_square_row_col(square):
      for i, x in enumerate(allsquares):
        for j, y in enumerate(x):
          if square == y: return i, j
      return None
    for i, s in enumerate(allsquares):
      for j, y in enumerate(s):
        tot = frozenset((battle[p[0]][p[1]] for p in y))
        points = set.union(*(groups[x] for x in tot))
        curstars = sum(1 for x in points if len(rem[x[0]][x[1]]) == 1 and 1 in rem[x[0]][x[1]])
        numstars = stars * len(tot) - curstars
        combs = get_mutex_regions(points, numstars)
        for comb in combs:
          ps = frozenset(y for y in comb if len(y) == 4)
          if len(comb) == numstars:
            for y in ps:
              z = get_square_row_col(y)
              if not z is None:
                if 0 in squaremap[z[0]][z[1]]: squaremap[z[0]][z[1]].remove(0)
        squares = get_square_regions(points)
        if not squares is None:
          squarerowcols = [get_square_row_col(x) for x in squares]
          if all(not x is None for x in squarerowcols):
            rowcols = {x[0] for x in squarerowcols}, {x[1] for x in squarerowcols}
            minrows = [(rc, [x[1] for x in squarerowcols if x[0] == rc]) for rc in rowcols[0]]
            mincols = [(rc, [x[0] for x in squarerowcols if x[1] == rc]) for rc in rowcols[1]]
            minr, minc = sum(len(x[1])-1 for x in minrows), sum(len(x[1])-1 for x in mincols)
            if minr == numstars:
              for x in minrows:
                for z in range(len(allsquares)):
                  if not z in x[1]:
                    if 0 in squaremap[x[0]][z]: squaremap[x[0]][z].remove(0)
            if minc == numstars:
              for x in mincols:
                for z in range(len(allsquares[0])):
                  if not z in x[1]:
                    if 0 in squaremap[z][x[0]]: squaremap[z][x[0]].remove(0)
    exclude = []
    for i, rc in enumerate(sum(1 for x in s if len(x) == 1) for s in squaremap):
      if rc == 4:
        if all(len(x) == 2 or 1 in x for x in squaremap[i]):
          r = next(iter(j for j, x in enumerate(squaremap[i]) if len(x) == 2))
          for p in allsquares[i][r]:
            if 1 in rem[p[0]][p[1]]: exclude.append((p[0], p[1], 1))
          squaremap[i][r].remove(1)
    for i, rc in enumerate(sum(1 for j in range(len(squaremap)) if len(squaremap[j][i]) == 1) for i in range(len(squaremap[0]))):
      if rc == 4:
        if all(len(x[i]) == 2 or 1 in x[i] for x in squaremap):
          r = next(iter(j for j, x in enumerate(squaremap) if len(x[i]) == 2))
          for p in allsquares[r][i]:
            if 1 in rem[p[0]][p[1]]: exclude.append((p[0], p[1], 1))
          squaremap[r][i].remove(1)
    if len(exclude) != 0: possible.append((exclude, BASIC_FISH, ()))
  return possible  
def shape_placement(rem, battle, stars, groups):
  possible, l = [], len(rem)
  regions = tuple(tuple(x) for x in standard_sudoku_singoverlap_regions(l))
  """
  for i in groups.keys():
    points = groups[i]
    curstars = sum(1 for x in points if len(rem[x[0]][x[1]]) == 1 and next(iter(rem[x[0]][x[1]])) == 1)
    numstars = stars - curstars
    points = {x for x in points if len(rem[x[0]][x[1]]) != 1}
    combs = get_mutex_regions(points, numstars)
    if combs == stars:
      if len(combs) != 0:
        for shape in combs:
          if len(shape) == 3:
            pts = tuple(shape)
            minx, maxx = min(pts, key=lambda p: p[0]), max(pts, key=lambda p: p[0])
            miny, maxy = min(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[1])
            for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
                if not (x, y) in shape: exclude.append((x, y, 1))
  """
  squarepoints = []
  for i in range(l-1):
    for j in range(l-1): #all combinations of connecting regions up to some length not just 2x2 ones should be considered
      tot = frozenset((battle[i][j], battle[i][j+1], battle[i+1][j], battle[i+1][j+1]))
      squarepoints.append((stars * len(tot), set.union(*(groups[x] for x in tot))))
  for allstars, points in [(stars, x) for region in regions for x in region] + [(stars, x) for x in groups.values()] + squarepoints:
    curstars = sum(1 for x in points if len(rem[x[0]][x[1]]) == 1 and 1 in rem[x[0]][x[1]])
    numstars = allstars - curstars
    points = {x for x in points if len(rem[x[0]][x[1]]) != 1}
    combs = get_mutex_regions(points, numstars)
    if len(combs) != 0: #point forcing
      exclude = []
      p = frozenset(y for x in combs for y in x if len(y) == 1)
      for s in p:
        if all(s in q for q in combs):
          if 0 in rem[next(iter(s))[0]][next(iter(s))[1]]: exclude.append((next(iter(s))[0], next(iter(s))[1], 0))
      pd = frozenset(y for x in combs for y in x if len(y) == 2)
      pnondiag = {domino for domino in pd if len(frozenset.intersection(*(king_rule_points(pt[0], pt[1], l)[0] for pt in domino))) == 4}
      for shape in pnondiag: #check rows/columns for 1 star and domino exclusion
        pts = tuple(shape)
        if pts[0][0] == pts[1][0]:
          starsonrowcol = tuple(x for x in range(l) if len(rem[pts[0][0]][x]) == 1 and 1 in rem[pts[0][0]][x])
          if len(starsonrowcol) == 1:
            for x in range(l):
              if x != starsonrowcol[0] and x != pts[0][1] and x != pts[1][1]:
                if 1 in rem[pts[0][0]][x]:
                  exclude.append((pts[0][0], x, 1))
        else:
          starsonrowcol = tuple(x for x in range(l) if len(rem[x][pts[0][1]]) == 1 and 1 in rem[x][pts[0][1]])
          if len(starsonrowcol) == 1:
            for x in range(l):
              if x != starsonrowcol[0] and x != pts[0][0] and x != pts[1][0]:
                if 1 in rem[x][pts[0][1]]:
                  exclude.append((x, pts[0][1], 1))
      if len(exclude) != 0: possible.append((exclude, X_CHAIN, ()))
      exclude = []
      if len(pd) != 0: #domino exclusions
        for x, y in frozenset.union(*(frozenset.intersection(*(z for q in shape for z in king_rule_points(q[0], q[1], l))) for shape in pd)):
          if 1 in rem[x][y]: exclude.append((x, y, 1))
      if len(exclude) != 0: possible.append((exclude, X_CHAIN, ()))
      exclude = []
      pt = frozenset(y for x in combs for y in x if len(y) == 3)
      if len(pt) != 0: #tromino exclusions
        for x, y in frozenset.union(*(frozenset.intersection(*(z for q in shape for z in king_rule_points(q[0], q[1], l))) for shape in pt)):
          if 1 in rem[x][y]: exclude.append((x, y, 1))
      if len(exclude) != 0: possible.append((exclude, X_CHAIN, ()))
      def place_stars(mutexes, comb, placements):
        if len(comb) == 0: return [placements] if len(placements) != 0 else []
        allplace = []
        for point in comb[0]:
          if any(ct == stars for m, ct in mutexes if point in m): continue
          allplace.extend(place_stars([(m, ct + (1 if point in m else 0)) for m, ct in mutexes], comb[1:], placements + [point]))
        return allplace
      mutexes = [row_points(i, 0, l) for i in range(l)] + [column_points(0, i, l) for i in range(l)] + [frozenset(i) for i in groups.values()]
      mutexes = [(x, sum(1 for p in x if len(rem[p[0]][p[1]]) == 1 and 1 in rem[p[0]][p[1]])) for x in mutexes]
      allplace = []
      for comb in combs: allplace.extend(place_stars(mutexes, comb, []))
      exclusions = frozenset() if len(allplace) == 0 else frozenset.intersection(*(frozenset.union(*(frozenset.union(*king_rule_points(y[0], y[1], l)) for y in x)) for x in allplace))
      exclude = []
      for x, y in exclusions:
        if 1 in rem[x][y]:
          exclude.append((x, y, 1))
      if len(exclude) != 0: possible.append((exclude, X_CHAIN, ()))
      if len(possible) != 0: break
  return possible
def x_chain_star_battle(rem, battle, stars, groups):
  l, possible = len(rem), []
  jigsaw_visibility = jigsaw_points_gen(groups.values())
  def x_chain_rcrse(rem, max_depth, chain, exclusions):
    x, y = chain[-1]
    if max_depth == len(chain) - 1: return possible
    exclusions |= frozenset(((x, y),))
    exclusions |= frozenset.union(*king_rule_points(x, y, l))
    for points in (*row_col_points(x, y, l), *jigsaw_visibility(x, y, l)):
      s = [p for p in points if 1 in rem[p[0]][p[1]] and len(rem[p[0]][p[1]]) == 1]
      if len(s) + len((points | frozenset(((x, y),))).intersection(chain)) == stars:
        exclusions |= points
    for points in (*(y for x in standard_sudoku_singoverlap_regions(l) for y in x), *groups.values()):
      rempoints = frozenset(points) - exclusions
      if len(rempoints) == 0: continue
      s = [p for p in rempoints if 1 in rem[p[0]][p[1]] and len(rem[p[0]][p[1]]) == 2]
      if len(s) == 0: #contradiction
        print(chain, exclusions, points, rempoints)
        return False
      elif len(s) == 1: #forcing link
        if not x_chain_rcrse(rem, max_depth, chain + [s[0]], exclusions): return False
    return True
  for x in range(l):
    for y in range(l):
      if len(rem[x][y]) != 1:
        if not x_chain_rcrse(rem, None, [(x, y)], frozenset()):
          possible.append((((x, y, 1),), X_CHAIN, ()))
  return possible
def exclude_star_battle(rem, battle, stars, groups):
  for func in (naked_single_star_battle, naked_multiples_star_battle, hidden_multiples_star_battle,
               region_space, bounded_regions, locked_candidates_star_battle, square_reduction, shape_placement,
               x_chain_star_battle):
    possible = func(rem, battle, stars, groups)
    if len(possible) != 0:
      #print(logic_step_string(possible[0], mutex_rules))
      for (x, y, z) in possible[0][0]:
        if not z in rem[x][y]:
          print(x, y, z, possible)
          print_candidate_border(rem, exc_from_border(battle, set()))
          raise ValueError
        else: rem[x][y].remove(z)
      #print_logic_step(possible[0])
      return rem, possible[0] 
  return rem, None
def star_battle_loop(rem, battle, stars):
  solve_path = []
  groups = jigsaw_to_coords(battle)
  while True:
    rem, found = exclude_star_battle(rem, battle, stars, groups)
    if found is None: break
    solve_path.append(found)
    if any(any(len(y) == 0 for y in x) for x in rem): return None, solve_path
  return rem, solve_path

def solve_star_battle(battle, stars):
  l = len(battle)
  rem = [[set((0, 1)) for _ in range(l)] for _ in range(l)]
  #groups = jigsaw_to_coords(battle)
  rem = star_battle_loop(rem, battle, stars)
  if rem is None: print("Bad Star Battle")
  return rem

def get_star_battles():
  normal_star_battle = ( #https://www.puzzle-star-battle.com/
    (1, 2, 2, 2, 2),
    (1, 1, 3, 4, 4),
    (1, 1, 3, 4, 4),
    (1, 1, 5, 4, 4),
    (1, 1, 5, 5, 4))
  normal_6x6_star_battle = ( #https://www.puzzle-star-battle.com/?size=1
    (1, 1, 1, 2, 2, 2),
    (3, 3, 3, 3, 2, 4),
    (5, 6, 6, 6, 2, 4),
    (5, 6, 6, 6, 2, 4),
    (5, 5, 6, 6, 6, 4),
    (5, 5, 6, 6, 6, 4))
  us_puzzle_championship_2019_star_battle = ( #https://www.youtube.com/watch?v=JQQeTht1KxM
    (1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
    (1, 1, 1, 2, 2, 2, 2, 3, 2, 2),
    (1, 1, 3, 3, 3, 3, 3, 3, 4, 4),
    (1, 1, 3, 5, 5, 5, 5, 3, 5, 4),
    (1, 6, 6, 6, 7, 7, 5, 5, 5, 4),
    (1, 8, 6, 7, 7, 7, 7, 7, 5, 4),
    (1, 8, 6, 7, 7, 7, 7, 5, 5, 4),
    (1, 8, 6, 7, 7, 9, 9, 5, 10, 4),
    (8, 8, 6, 9, 7, 9, 9, 10, 10, 10),
    (8, 8, 6, 9, 9, 9, 9, 9, 10, 10))
  lm_star_battle = ( #https://logicmastersindia.com/forum/forums/thread-view.asp?tid=140
    (1, 1, 1, 1, 2, 2, 2, 2, 2),
    (1, 1, 2, 2, 2, 2, 3, 2, 2),
    (1, 1, 1, 3, 3, 3, 3, 4, 4),
    (1, 5, 5, 6, 3, 3, 3, 4, 4),
    (5, 5, 5, 6, 6, 3, 3, 3, 4),
    (5, 5, 5, 6, 6, 6, 4, 4, 4),
    (5, 7, 5, 5, 8, 4, 4, 9, 4),
    (7, 7, 7, 8, 8, 8, 9, 9, 9),
    (7, 7, 7, 8, 8, 8, 9, 9, 9))
  easier_star_battle = ( #https://www.youtube.com/watch?v=oUkFRnR2Z-g
    (1, 1, 1, 2, 2, 2, 2, 2, 2, 2),
    (1, 2, 2, 2, 3, 4, 5, 5, 5, 5),
    (1, 2, 1, 1, 3, 4, 5, 5, 5, 5),
    (1, 1, 1, 1, 3, 4, 5, 5, 5, 6),
    (1, 3, 3, 3, 3, 4, 4, 4, 4, 6),
    (1, 7, 7, 7, 7, 8, 8, 8, 8, 6),
    (1, 1, 1, 10, 7, 8, 10, 10, 6, 6),
    (11, 11, 1, 10, 7, 8, 10, 10, 6, 6),
    (11, 11, 1, 10, 7, 8, 10, 10, 10, 10),
    (11, 11, 11, 10, 10, 10, 10, 10, 10, 10))
  perfect_star_battle = ( #https://www.youtube.com/watch?v=xQbv83aUHLU
   (1, 1, 1, 1, 1, 1, 1, 1, 2, 2),
   (1, 3, 3, 3, 3, 3, 2, 2, 2, 2),
   (3, 3, 4, 5, 5, 5, 2, 2, 6, 6),
   (3, 3, 4, 5, 5, 5, 6, 6, 6, 6),
   (3, 7, 4, 4, 4, 5, 5, 5, 6, 6),
   (8, 7, 4, 4, 4, 4, 4, 5, 6, 6),
   (8, 7, 7, 7, 4, 5, 5, 5, 6, 9),
   (8, 7, 7, 7, 4, 4, 4, 5, 10, 9),
   (8, 7, 8, 7, 10, 10, 10, 10, 10, 9),
   (8, 8, 8, 10, 10, 9, 9, 9, 9, 9))
  return ((normal_star_battle, 1),
          (normal_6x6_star_battle, 1),
          (us_puzzle_championship_2019_star_battle, 2),
          (lm_star_battle, 2),
          (easier_star_battle, 2),
          (perfect_star_battle, 2),)
def valid_star_battle(rem, battle, stars):
  if rem is None: return rem, False
  l = len(rem)
  if any(any(len(y) != 1 for y in x) for x in rem): return rem, False
  rem = [[next(iter(x)) for x in y] for y in rem]
  for points in star_battle_regions(l, battle):
    if any(sum(rem[p[0]][p[1]] for p in points) != stars for x in range(l)): return rem, False
  return rem, True
def check_star_battle(battle):
  l = len(battle[0])
  rem, solve_path = solve_star_battle(battle[0], battle[1])
  rem, valid = valid_star_battle(rem, battle[0], battle[1])
  if not rem is None:
    print_border(rem, exc_from_border(battle[0], set()))
    if not valid:
      print_candidate_border(rem, exc_from_border(battle[0], set()))
def check_star_battles(battles):
  for y in battles: check_star_battle(y)

check_puzzle(get_ctc_puzzles()[0])
#check_puzzle(standard_sudoku(str_to_sudoku("...........19.2.6......679.9.2...6..37....95...5.....414...3..57.9.24......8....."))) #Skyscraper
#check_puzzle(standard_sudoku(str_to_sudoku(".81.2............9..68...4...31..7....8.57916........45....6.9223...84..........."))) #2-String Kite
#check_puzzle(standard_sudoku(str_to_sudoku("7...56.381.842.............5..3......4..8.7....1...24...3.........1....5.5...7.6."))) #Empty Rectangle
#check_puzzles(get_ctc_puzzles())
#check_puzzles(get_impossible_puzzles())
#check_puzzles(file_puzzles("topn87.txt"))

#check_star_battles(get_star_battles())

#import sys
#if sys.version_info.major == 3 and sys.version_info.minor == 7:
try:
  from manimlib.imports import *
except ModuleNotFoundError:
  class Scene:
    def __init__(): pass

class Sudoku(Scene):
  def draw_board(self, puzzle):
    sudoku = puzzle[0]
    mutex_rules = puzzle[1]
    cell_visibility_rules = puzzle[2]
    cell_visibility_rules = mutex_regions_to_visibility(mutex_rules) + cell_visibility_rules
    value_set = puzzle[4]
    l = len(sudoku)
    #squares = [[Square(fill_color=BLACK, fill_opacity=1, color=WHITE, side_length=0.5) for _ in range(l)] for _ in range(l)]
    #squares[4][4].move_to((0, 0, 0))
    #for i in range(l):
    #  for j in range(l):
    #    if i == 0 and j == 0: squares[i][j].move_to(DOWN+LEFT)
    #    elif j == 0: squares[i][j].move_to(squares[i-1][j].get_center() + (0, squares[i-1][j].get_height(), 0))
    #    else: squares[i][j].move_to(squares[i][j-1].get_center() + (squares[i][j-1].get_width(), 0, 0))
    #self.play(*(FadeIn(squares[i][j]) for i in range(l) for j in range(l)))
    #self.wait(10)
    #self.remove(*(squares[i][j] for i in range(l) for j in range(l)))
    cell_size = 0.8
    horlines = [[Line(start=BOTTOM+LEFT_SIDE+(j*cell_size, i*cell_size, 0), end=BOTTOM+LEFT_SIDE+(j*cell_size+cell_size, i*cell_size, 0), color=WHITE, stroke_width=4 if i % 3 == 0 else 2) for j in range(l)] for i in range(l+1)]
    verlines = [[Line(start=BOTTOM+LEFT_SIDE+(j*cell_size, i*cell_size, 0), end=BOTTOM+LEFT_SIDE+(j*cell_size, i*cell_size+cell_size, 0), color=WHITE, stroke_width=4 if j % 3 == 0 else 2) for j in range(l+1)] for i in range(l)]
    nums = []
    for i in range(l):
      for j in range(l):
        if not sudoku[i][j] is None:
          t = TextMobject(str(sudoku[i][j]))
          t.move_to(BOTTOM+LEFT_SIDE+(j*cell_size+cell_size/2, (l-1-i)*cell_size+cell_size/2, 0))
          nums.append(t)
    self.play(*(FadeIn(horlines[i][j]) for i in range(l+1) for j in range(l)),*(FadeIn(verlines[i][j]) for i in range(l) for j in range(l+1)), *(FadeIn(i) for i in nums))
    #self.wait(10)
    rem = init_sudoku(sudoku, cell_visibility_rules, value_set)
    _, solve_path, border = solve_sudoku(sudoku, mutex_rules, puzzle[2], puzzle[3], value_set, puzzle[5])
    smallnums = [[dict() for _ in range(l)] for _ in range(l)]
    smallnum_dist = ((2,), (3,), (2,2), (3,2), (3,3), (3,2,2), (3,3,2), (3,3,3))
    for i in range(l):
      for j in range(l):
        if len(rem[i][j]) != 1:
          d, line, dist = 0, 0, smallnum_dist[len(rem[i][j])-2]
          line_inc = cell_size / (len(dist) + 1)
          hor_inc = cell_size / (dist[line] + 1)
          for k in sorted(rem[i][j]):
            smallnums[i][j][k] = TextMobject(str(k))
            smallnums[i][j][k].move_to(BOTTOM+LEFT_SIDE+(j*cell_size+hor_inc*(d+1), (l-1-i)*cell_size+line_inc*(line+1), 0))
            smallnums[i][j][k].scale(0.3)
            self.add(smallnums[i][j][k])
            d += 1
            if d >= dist[line]:
              line += 1
              if line == len(dist): break
              d = 0
              hor_inc = cell_size / (dist[line] + 1)
    self.wait(10)
    print(len(solve_path))
    def scale_move_gen(i, j):
      def scale_move(x):
        x.scale(3.3333) #0.3*z=n, so z = n / 0.3 and 1/0.3=3.3333
        x.move_to(BOTTOM+LEFT_SIDE+(j*cell_size+cell_size/2, (l-1-i)*cell_size+cell_size/2, 0))
        return x
      return scale_move
    for x in solve_path:
      step = TextMobject(logic_step_string(x, mutex_rules, True))
      step.move_to(RIGHT * 3)
      self.add(step)
      removals = []
      anims = []
      postanims = []
      if x[1] in (LAST_DIGIT, FULL_HOUSE, NAKED_SINGLE):
        s = Square(side_length=cell_size, fill_color=GREEN, fill_opacity=0.3)
        s.move_to(BOTTOM+LEFT_SIDE+(x[2][0][1]*cell_size+cell_size/2, (l-1-x[2][0][0])*cell_size+cell_size/2, 0))
        anims.append(FadeIn(s))
        removals.append(s)
        for p in cell_visibility(x[2][0][0], x[2][0][1], l, cell_visibility_rules):
          sp = Square(side_length=cell_size, fill_color=BLUE, fill_opacity=0.3)
          sp.move_to(BOTTOM+LEFT_SIDE+(p[1]*cell_size+cell_size/2, (l-1-p[0])*cell_size+cell_size/2, 0))
          anims.append(FadeIn(sp))
          removals.append(sp)
      elif x[1] == HIDDEN_SINGLE:
        for p in x[2][0]:
          sp = Square(side_length=cell_size, fill_color=GREEN if p == (x[0][0][0], x[0][0][1]) else BLUE, fill_opacity=0.3)
          sp.move_to(BOTTOM+LEFT_SIDE+(p[1]*cell_size+cell_size/2, (l-1-p[0])*cell_size+cell_size/2, 0))
          anims.append(FadeIn(sp))
          removals.append(sp)
      elif x[1] == LOCKED_CANDIDATES:
        for p in x[2][0]:
          sp = Square(side_length=cell_size, fill_color=GREEN if p in x[2][2] else BLUE, fill_opacity=0.3)
          sp.move_to(BOTTOM+LEFT_SIDE+(p[1]*cell_size+cell_size/2, (l-1-p[0])*cell_size+cell_size/2, 0))
          anims.append(FadeIn(sp))
          removals.append(sp)
        for p in x[2][1]:
          sp = Square(side_length=cell_size, fill_color=RED, fill_opacity=0.3)
          sp.move_to(BOTTOM+LEFT_SIDE+(p[1]*cell_size+cell_size/2, (l-1-p[0])*cell_size+cell_size/2, 0))
          anims.append(FadeIn(sp))
          removals.append(sp)
        for p in x[2][2]:
          anims.append(ApplyMethod(smallnums[p[0]][p[1]][x[0][0][2]].set_color, GREEN))
          postanims.append(ApplyMethod(smallnums[p[0]][p[1]][x[0][0][2]].set_color, WHITE))          
      elif x[1] == HIDDEN_MULTIPLES or x[1] == NAKED_MULTIPLES:
        for p in x[2][2]:
          sp = Square(side_length=cell_size, fill_color=GREEN if p in x[2][0] else BLUE, fill_opacity=0.3)
          sp.move_to(BOTTOM+LEFT_SIDE+(p[1]*cell_size+cell_size/2, (l-1-p[0])*cell_size+cell_size/2, 0))
          anims.append(FadeIn(sp))
          removals.append(sp)
        for p in x[2][0]:
          for y in x[2][1]:
            if y in smallnums[p[0]][p[1]]:
              anims.append(ApplyMethod(smallnums[p[0]][p[1]][y].set_color, GREEN))
              postanims.append(ApplyMethod(smallnums[p[0]][p[1]][y].set_color, WHITE))
      elif x[1] == BASIC_FISH or x[1] == FINNED_FISH:
        matches = frozenset.union(*x[2][1])
        for p in x[2][0]:
          sp = Square(side_length=cell_size, fill_color=GREEN if p in matches else BLUE, fill_opacity=0.3)
          sp.move_to(BOTTOM+LEFT_SIDE+(p[1]*cell_size+cell_size/2, (l-1-p[0])*cell_size+cell_size/2, 0))
          anims.append(FadeIn(sp))
          removals.append(sp)
        for p in x[2][2]:
          sp = Square(side_length=cell_size, fill_color=RED, fill_opacity=0.3)
          sp.move_to(BOTTOM+LEFT_SIDE+(p[1]*cell_size+cell_size/2, (l-1-p[0])*cell_size+cell_size/2, 0))
          anims.append(FadeIn(sp))
          removals.append(sp)
        for p in matches:
          anims.append(ApplyMethod(smallnums[p[0]][p[1]][x[0][0][2]].set_color, GREEN))
          postanims.append(ApplyMethod(smallnums[p[0]][p[1]][x[0][0][2]].set_color, WHITE))
      elif x[1] == X_CHAIN:
        for i, p in enumerate(x[2][0]):
          anims.append(ApplyMethod(smallnums[p[0]][p[1]][x[0][0][2]].set_color, GREEN if i & 1 == 1 and x[2][1] != 1 or x[2][1] == 0 or x[2][1] == 1 and (i == 0 or i == 3) else BLUE))
          postanims.append(ApplyMethod(smallnums[p[0]][p[1]][x[0][0][2]].set_color, WHITE))          
      for exclude in x[0]:
        anims.append(ApplyMethod(smallnums[exclude[0]][exclude[1]][exclude[2]].set_color, RED))
        removals.append(smallnums[exclude[0]][exclude[1]][exclude[2]])
        smallnums[exclude[0]][exclude[1]].pop(exclude[2])
        if len(smallnums[exclude[0]][exclude[1]]) == 1:
          k = next(iter(smallnums[exclude[0]][exclude[1]]))
          postanims.append(ApplyFunction(scale_move_gen(exclude[0], exclude[1]), smallnums[exclude[0]][exclude[1]][k]))
      if len(anims) != 0: self.play(*anims)
      self.wait(2)
      self.remove(*removals)
      if len(postanims) != 0: self.play(*postanims)
      self.remove(step)
    self.remove(*(horlines[i][j] for i in range(l+1) for j in range(l)), *(verlines[i][j] for i in range(l) for j in range(l+1)), *nums)
  def construct(self):
    self.draw_board(get_ctc_puzzles()[0])
    
