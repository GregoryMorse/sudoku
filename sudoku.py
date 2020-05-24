#import os; exec(open(os.path.join('D:', 'OneDrive', 'Documents', 'Projects', 'sudoku.py')).read())
import itertools

def orthagonal_pts(i, j): return ((i, j-1), (i-1, j), (i+1, j), (i, j+1))
def orthagonal_points(i, j, l):
  return (frozenset(filter(lambda x: x[0] >= 0 and x[0] <= l-1 and x[1] >= 0 and x[1] <= l-1, orthagonal_pts(i, j))),)
def diagonal_pts(i, j): return ((i-1, j-1), (i+1, j-1), (i-1, j+1), (i+1, j+1))
def king_rule_pts(i, j): return ((i-1, j-1), (i, j-1), (i+1, j-1), (i-1, j), (i+1, j), (i-1, j+1), (i, j+1), (i+1, j+1))
def king_rule_points(i, j, l):
  return (frozenset(filter(lambda x: x[0] >= 0 and x[0] <= l-1 and x[1] >= 0 and x[1] <= l-1, king_rule_pts(i, j))),)
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
  return (frozenset(filter(lambda x: x[0] >= 0 and x[0] <= l-1 and x[1] >= 0 and x[1] <= l-1, ((i-2, j-1), (i-2, j+1), (i-1, j-2), (i-1, j+2), (i+1, j-2), (i+1, j+2), (i+2, j-1), (i+2, j+1)))),)
def row_points(i, j, l): return frozenset((i, x) for x in range(l))
def column_points(i, j, l): return frozenset((x, j) for x in range(l))
def subsquare_points(i, j, l): return frozenset(get_sub_square_points(sub_square_from_point(i, j, l), l))
def diagonal_points(i, j, l):
  return frozenset.union(frozenset((x, x) for x in range(l)) if i == j else frozenset(),
                         frozenset((x, l-1-x) for x in range(l)) if i == l - 1 - j else frozenset())
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
def cell_mutex_visibility(i, j, l, cell_visibility_rules):
  return (x - frozenset(((i, j),)) for c in cell_visibility_rules for x in c(i, j, l))
def cell_sudoku_rule(rem, i, j, y, cell_visibility_rules):
  #print("Initialization via Full House/Naked Single %d to (%d, %d)" % (y, i, j))
  l = len(rem)
  for (p, q) in cell_visibility(i, j, l, cell_visibility_rules):
    if y in rem[p][q]: rem[p][q].remove(y)
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
def check_bad_sudoku(rem):
  if any(any(len(y) == 0 for y in x) for x in rem): return True
  return False
LAST_DIGIT,FULL_HOUSE,NAKED_SINGLE,HIDDEN_SINGLE,LOCKED_CANDIDATES,NAKED_MULTIPLES,HIDDEN_MULTIPLES,BASIC_FISH,FINNED_FISH,X_CHAIN,XY_CHAIN=0,1,2,3,4,5,6,7,8,9,10
KILLER_CAGE_RULE,THERMO_RULE,INEQUALITY_RULE,HIDDEN_CAGE_TUPLE,MIRROR_CAGE_CELL,ORTHAGONAL_NEIGHBOR,MAGIC_SQUARE,SANDWICH_RULE,ARROW_RULE=11,12,13,14,15,16,17,18,19
STRING_RULES = ("Last Digit", "Full House", "Naked Single", "Hidden Single", "Locked Candidates", "Naked %s", "Hidden %s", "Basic Fish", "Finned Fish", "X-Chain", "XY-Chain",
                "Killer Cage Rule", "Thermometer Rule", "Inequality Rule", "Hidden Cage Tuple", "Mirror Cage Cell", "Orthagonal Neighbor", "Magic Square", "Sandwich Rule", "Arrow Rule")
def naked_single(rem, mutex_rules, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  #should find the full houses before naked singles followed by hidden singles
  for i in range(l):
    for j in range(l):
      if len(rem[i][j]) != 1: continue
      y = next(iter(rem[i][j]))
      t = tuple(x for x in cell_visibility(i, j, l, cell_visibility_rules) if y in rem[x[0]][x[1]])
      if len(t) != 0: #[(i, j, y) for y in rem[i][j].intersection(vals)]
        classify = NAKED_SINGLE
        for region in cell_mutex_visibility(i, j, l, cell_visibility_rules):
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
def hidden_single(rem, mutex_rules, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  regions = tuple(r for regs in mutex_rules for r in regs)
  for y in value_set:
    for points in regions:
      exclusive = tuple(z for z in points if y in rem[z[0]][z[1]])
      if len(exclusive) == 1:
        if rem[exclusive[0][0]][exclusive[0][1]] == set((y,)): continue
        possible.append(([(exclusive[0][0], exclusive[0][1], z) for z in rem[exclusive[0][0]][exclusive[0][1]] - set((y,))], HIDDEN_SINGLE, (points,)))
  return possible
def locked_candidates(rem, mutex_rules, cell_visibility_rules, value_set):
  l, possible = len(rem), []
  for points in (r for regs in mutex_rules for r in regs):
    for y in value_set:
      pts = tuple(z for z in points if y in rem[z[0]][z[1]])
      a = tuple(cell_visibility(z[0], z[1], l, cell_visibility_rules) for z in pts)
      s = frozenset.intersection(*a) if len(a) != 0 else frozenset()
      exclude = []
      for o, z in s.difference(points):
        if y in rem[o][z]:
          #if points is a row/column then it is claiming, otherwise it is pointing and pointing either on row/column depending on if o or z matches up with any row/column in points
          exclude.append((o, z, y))
      if len(exclude) != 0: possible.append((exclude, LOCKED_CANDIDATES, (points,s,pts)))
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
            for r in frozenset.union(*opp).difference(su):
              if y in rem[r[0]][r[1]]:
                exclude.append((r[0], r[1], y))
            if len(exclude) != 0: possible.append((exclude, BASIC_FISH, (region, s, y, mutex)))
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
                for r in subs.intersection(frozenset.union(*opp).difference(su).difference(s[j])):
                  if y in rem[r[0]][r[1]]:
                    exclude.append((r[0], r[1], y))
                if len(exclude) != 0: possible.append((exclude, FINNED_FISH, (region, s, y, mutex)))
  return possible
  
def x_chain(rem, mutex_rules, cell_visibility_rules, value_set, max_depth): #solves skyscrapers, empty rectangles (with or without 2 candidates), turbot fish, 2-string kites
  #dual 2-kite strings and dual empty rectangles would not be explicitly identified and are the combination of 2 X-chains
  l, possible = len(rem), []
  def x_chain_rcrse(rem, max_depth, chain, search, exclusions):
    possible = []
    x, y = chain[-1]
    if max_depth == len(chain): return possible
    for points in cell_mutex_visibility(x, y, l, cell_visibility_rules):
      s = [p for p in points if search in rem[p[0]][p[1]]]
      if len(s) == 1 and len(exclusions.intersection(s)) == 0: #strong link
        if len(chain) != 1:
          exclude = []
          for p, q in frozenset.intersection(frozenset(cell_visibility(s[0][0], s[0][1], l, cell_visibility_rules)), frozenset(cell_visibility(chain[0][0], chain[0][1], l, cell_visibility_rules))):
            if not (p, q) in chain and (p, q) != s[0] and search in rem[p][q]:
              #4-length X chain can be a skyscraper or empty rectangle
              #print("X-Chain %d in (%d, %d) of length %d" % (search, p, q, len(chain)+1) + " " + str(chain))
              exclude.append((p, q, search))
          if len(exclude) != 0: possible.append((exclude, X_CHAIN, (chain + [s[0]], search)))
        cv = cell_visibility(s[0][0], s[0][1], l, cell_visibility_rules)
        for point in cv:
          if point in chain or not search in rem[point[0]][point[1]]: continue
          possible.extend(x_chain_rcrse(rem, max_depth, chain + [s[0], point], search, exclusions.union(cv)))
    return possible
  for x in range(l):
    for y in range(l):
      if len(rem[x][y]) != 1:
        for z in rem[x][y].copy():
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
def logic_step_string(step):
  multiples = ("Pairs", "Triples", "Quadruples", "Quintuples", "Sextuples", "Septuples", "Octuples", "Nonuples", "Decuple")
  name = STRING_RULES[step[1]]
  if step[1] in (LAST_DIGIT, FULL_HOUSE, NAKED_SINGLE):
    logic = "in " + str(step[2][0])
  elif step[1] == HIDDEN_SINGLE:
    logic = "along row/column/region " + str(step[2][0])
  elif step[1] == LOCKED_CANDIDATES:
    logic = "in region " + str(step[2][0]) + " cells " + str(step[2][2]) + " pointing at row/column " + str(step[2][1])
    logic = "in row/column " + str(step[2][0]) + " cells " + str(step[2][2]) + " claiming from region " + str(step[2][1])
  elif step[1] == HIDDEN_MULTIPLES:
    name = name % multiples[len(step[2][0])-1-1]
    logic = "in row/column/region " + str(step[2][2]) + " cells " + str(step[2][0]) + " with values " + str(sorted(step[2][1]))
  elif step[1] == NAKED_MULTIPLES:
    name = name % multiples[len(step[2][0])-1-1] 
    logic = "in row/column/region " + str(step[2][2]) + " cells " + str(step[2][0]) + " with values " + str(sorted(step[2][1]))
  else: logic = str(step[2])
  return name + " " + logic + " excludes " + ", ".join("%d in (%d, %d)" % (z, x, y) for (x, y, z) in step[0])
  
def logical_solve_string(steps):
  return '\n'.join(logic_step_string(x) for x in steps)

def exclude_sudoku_by_group(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set): #combine rule for row/column with sub regions
  if check_bad_sudoku(rem): return rem, False
  for func in exclude_rule + (naked_single, hidden_single, locked_candidates, mutex_multiples, basic_fish, finned_fish, #skyscraper, empty_rectangle,
               lambda rm, mr, cv, vs: x_chain(rm, mr, cv, vs, 4), lambda rm, mr, cv, vs: xy_chain(rm, mr, cv, vs, 3),
               lambda rm, mr, cv, vs: x_chain(rm, mr, cv, vs, 6), lambda rm, mr, cv, vs: xy_chain(rm, mr, cv, vs, None)):
    possible = func(rem, mutex_rules, cell_visibility_rules, value_set)
    if len(possible) != 0:
      for (x, y, z) in possible[0][0]:
        if not z in rem[x][y]:
          print(x, y, z, possible)
          #raise ValueError
        else: rem[x][y].remove(z)
      #print_logic_step(possible[0])
      return rem, possible[0]
  return rem, None

def sudoku_loop(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set):
  solve_path = []
  while True:
    rem, found = exclude_sudoku_by_group(rem, mutex_rules, cell_visibility_rules, exclude_rule, value_set)
    if any(any(len(y) == 0 for y in x) for x in rem):
      #print_candidate_format(rem)
      return None, solve_path
    if found is None: break
    solve_path.append(found)
  return rem, solve_path

#len(list(itertools.permutations((1,2,3,4,5,6,7,8,9),9))) == 362880 permutations of numbers 1 to 9
#for 9x9 classic sudoku: 6670903752021072936960 possible grids 5472730538 essentially unique solutions
def solve_sudoku(sudoku, mutex_rules, cell_visibility_rules, exclude_rule, value_set, border_solve):
  l = len(sudoku)
  if not border_solve is None:
    border = border_solve()
    mutex_rules = (*mutex_rules, tuple(frozenset(x) for x in jigsaw_to_coords(border).values()))
  else: border = None
  cell_visibility_rules = mutex_regions_to_visibility(mutex_rules) + cell_visibility_rules
  rem = [[set(value_set) for _ in range(l)] for _ in range(l)]
  for i, x in enumerate(sudoku):
    for j, y in enumerate(x):
      if not y is None:
        rem = cell_sudoku_rule(rem, i, j, y, cell_visibility_rules)
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

#this rule does not guarantee uniqueness unless it continues searching
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
  rem = [[0 for _ in range(l)] for _ in range(l)]
  for i, x in enumerate(killer):
    for z in x[1]:
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
def get_all_mutex_digit_sum(total, digits, value_set):
  combs = []
  for comb in itertools.combinations(value_set, digits):
    if sum(comb) == total: combs.append(comb)
  return combs
def get_all_mutex_digit_sum_rcrse(total, digits, values): #more efficient due to exclusions
  def get_all_mutex_digit_sum_inner(total, digits, values):
    if digits == 1: return [(total,)] if total in values else []
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
    l = []
    for y in value_sets[0]:
      if not y in used_values:
        l.extend([(y, *x) for x in get_all_mutex_digit_sum_inner(total - y, value_sets[1:], used_values | set((y,)))])
    return l
  return get_all_mutex_digit_sum_inner(total, value_sets, used_values)
def has_mutex_digit_sum(total, value_sets, used_values):
  #get_all_mutex_digit_sum_sets(43, [[x for x in range(2, 10)] for _ in range(7)], set((1,)))
  def has_mutex_digit_sum_inner(total, value_sets, used_values):
    if len(value_sets) == 1: return True if total in value_sets[0] and not total in used_values else False
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
def exclude_killer_rule_gen(killer): #digits unique in a cage
  exclude_jigsaw_rule = exclude_jigsaw_rule_gen([y for _, y in killer])
  def exclude_killer_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    tot = sum(z for z in value_set)
    for x, y in killer:
      exclude = []
      for i in range(len(y)):
        if len(rem[y[i][0]][y[i][1]]) == 1: continue
        for val in rem[y[i][0]][y[i][1]]:
          if not has_mutex_digit_sum(x - val, [rem[y[j][0]][y[j][1]] for j in range(len(y)) if j != i], set((val,))): #len(get_all_mutex_digit_sum_sets()) == 0
            exclude.append((y[i][0], y[i][1], val))
      if len(exclude) != 0: possible.append((exclude, KILLER_CAGE_RULE, (x, y)))
      #can exclude any sets where a double interferes with a 2 visible triples, a triple interferes with a 2/3 visible quadruples, or 3 quintuples, a quadruple interferes with a 4 quintuples
      if len(y) == 2:
        exclude = []
        sets = get_all_mutex_digit_sum_sets(x, [rem[y[j][0]][y[j][1]] for j in range(len(y))], set())
        removesets = set()
        for regions in mutex_rules:
          for region in regions:
            intsct = region.intersection(y)
            if len(intsct) >= 2:
              for z in sets:
                vals = frozenset((z[y.index(p)] for p in intsct))
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
      if len(exclude) != 0: possible.append((exclude, HIDDEN_CAGE_TUPLE, ()))
    return possible
  return exclude_cage_hidden_tuple_rule
def exclude_cage_mirror_rule_gen(cages):
  def exclude_cage_mirror_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    exclude = []
    for c in cages:
      for i in range(len(c)):
        a = tuple(cell_visibility(x, y, l, cell_visibility_rules) for x, y in c[:i] + c[i+1:] if len(rem[x][y]) != 1)
        visible = frozenset.intersection(*a) if len(a) != 0 else frozenset()
        visible = {x for x in visible if len(rem[x[0]][x[1]]) != 1}
        if len(visible) == 1 and len(visible.intersection(cell_visibility(c[i][0], c[i][1], l, cell_visibility_rules))) == 0:
          p = next(iter(visible))
          for y in rem[c[i][0]][c[i][1]] - rem[p[0]][p[1]]:
            exclude.append((c[i][0], c[i][1], y))
    if len(exclude) != 0: possible.append((exclude, MIRROR_CAGE_CELL, ()))
    return possible
  return exclude_cage_mirror_rule
def exclude_sandwich_rule_gen(sandwiches):
  def exclude_sandwich_rule(rem, mutex_rules, cell_visibility_rules, value_set):
    l, possible = len(rem), []
    tot = sum(value_set)
    mins, maxes = [], [] #mins nearest rounded down, maxes nearest rounded up
    sandwichset = set((min(value_set), max(value_set))) #1 and 9
    for x in sorted(value_set):
      if len(mins) == 0: mins.append(x)
      else: mins.append(mins[-1] + x)
    for x in reversed(sorted(value_set)):
      if len(maxes) == 0: maxes.append(x)
      else: maxes.append(maxes[-1] + x)
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
  #print(logical_solve_string(solve_path))
  if not rem is None:
    y[6][0](rem) if y[5] is None else y[6][0](rem, border)
    if not valid:
      #y[6][0](brute_sudoku_depth(rem, y[2], y[3]))
      y[6][1](rem) if y[5] is None else y[6][0](rem, border)
  else: print("Bad Sudoku")

def check_puzzles(puzzles):
  for y in puzzles: check_puzzle(y)

def str_to_sudoku(s):
  sudoku = [[[] for _ in range(9)] for _ in range(9)]
  for i in range(len(s)):
    sudoku[i // 9][i % 9] = int(s[i]) if s[i] != '.' else None
  return sudoku

def standard_sudoku(puzzle):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def king_sudoku(puzzle):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (king_rule_points,), (), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
def knight_sudoku(puzzle):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (knight_rule_points,), (), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
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
def thermo_sudoku(puzzle, thermo):
  l = len(puzzle)
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_thermo_rule_gen(thermo),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format))
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
def killer_cages_sudoku(killer, l):
  #tuple(frozenset(x) for x in jigsaw_to_coords(killer_to_jigsaw(killer, l)).values()) #not mutual exclusive across value set
  #if killer cage is in a row, column or subsquare we dont need to include it as part of cell visibility since its redundant...
  return (((None,) * l,) * l, standard_sudoku_mutex_regions(l), (jigsaw_points_gen(killer_to_jigsaw(killer, l)),), (exclude_killer_rule_gen(killer), exclude_cage_hidden_tuple_rule_gen([y for _, y in killer]), exclude_cage_mirror_rule_gen([y for _, y in killer])), frozenset(range(1, l+1)), None,
          (lambda x: print_border(x, exc_from_border(killer_to_jigsaw(killer, l), set())), lambda x: print_candidate_border(x, exc_from_border(killer_to_jigsaw(killer, l), set()))))
def sandwich_sudoku(puzzle, sandwich_row_cols):
  l = len(puzzle)
  sandwiches = [(x, row_points(i, 0, l)) for i, x in enumerate(sandwich_row_cols[0]) if not x is None] + [(x, column_points(0, i, l)) for i, x in enumerate(sandwich_row_cols[1]) if not x is None]
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_sandwich_rule_gen(sandwiches),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format)) 
def sandwich_arrow_sudoku(puzzle, arrows, increasing, sandwich_row_cols):
  l = len(puzzle)
  sandwiches = [(x, row_points(i, 0, l)) for i, x in enumerate(sandwich_row_cols[0]) if not x is None] + [(x, column_points(0, i, l)) for i, x in enumerate(sandwich_row_cols[1]) if not x is None]
  return (puzzle, standard_sudoku_mutex_regions(l), (), (exclude_arrow_rule_gen(arrows, increasing), exclude_sandwich_rule_gen(sandwiches),), frozenset(range(1, l+1)), None, (print_sudoku, print_candidate_format)) 
  
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
  
  killer_xxl_sudoku = ( #https://www.youtube.com/watch?v=qQ-B8R3wnEM&t=1055s
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
    
  excellence_elegance_sudoku = [[None] * 9 for _ in range(9)]
  excellence_elegance_arrows = (
    ((3, 3), (2, 2), (1, 1)),
    ((3, 5), (2, 6), (1, 7), (0, 8)),
    ((5, 3), (6, 2), (7, 1), (8, 0)),
    ((5, 5), (6, 6), (7, 7), (8, 8)))
  excellence_elegance_sandwich_row_cols = ((20, None, None, 20, 33, None, 0, None, None), (None, None, None, 27, None, 13, None, None, 13))    
  
  #print_border(((None,) * 6,) * 6, prize_sudoku_subsquare_exc)
  #print_border(get_border_count([[None] * 6 for _ in range(6)], prize_sudoku_subsquare_exc), prize_sudoku_subsquare_exc)
  #print_border(add_region([[None] * 6 for _ in range(6)], max_region(region_ominos((3, 3), [[None] * 6 for _ in range(6)], prize_sudoku_subsquare_exc))[0], 1), prize_sudoku_subsquare_exc)
  #print_border(brute_border(prize_sudoku_subsquare_exc, 6)[1], exc_from_border(brute_border(prize_sudoku_subsquare_exc, 6)[1], prize_sudoku_subsquare_exc))
  #print_candidate_border(brute_border(prize_sudoku_subsquare_exc, 6)[0], prize_sudoku_subsquare_exc)
  #print_border(brute_border(nightmare_sudoku_subsquare_exc, 6)[1], exc_from_border(brute_border(nightmare_sudoku_subsquare_exc, 6)[1], nightmare_sudoku_subsquare_exc))
  #print_candidate_border(brute_border(nightmare_sudoku_subsquare_exc, 6)[0], nightmare_sudoku_subsquare_exc)
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
    arrow_sudoku(fibo_series_two_sudoku, fibo_series_two_arrows, False),
    sandwich_arrow_sudoku(excellence_elegance_sudoku, excellence_elegance_arrows, True, excellence_elegance_sandwich_row_cols),
    sandwich_sudoku(slippery_sandwich_sudoku, slippery_sandwich_row_cols),
    knight_thermo_magic_square_sudoku(aad_tribute_sudoku, aad_tribute_thermo, aad_magic_squares),
    king_knight_magic_square_sudoku(magical_miracle_sudoku, magical_miracle_squares),
    king_knight_orthagonal_sudoku(miracle_sudoku),
    thermo_sudoku(thermo_app_sudoku, thermo_app_sudoku_thermometers),
    king_sudoku(pi_king_sudoku),
    knight_sudoku(us_championship_knights_sudoku),
    knight_sudoku(expert_knights_sudoku),
    jigsaw_sudoku(prize_sudoku, prize_sudoku_jigsaw),
    partial_border_jigsaw_sudoku(prize_sudoku, prize_sudoku_subsquare_exc),
    partial_border_diagonal_thermo_sudoku(nightmare_sudoku, nightmare_sudoku_subsquare_exc, nightmare_sudoku_thermo),
    killer_cages_sudoku(killer_xxl_sudoku, 9),
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
def cell_star_battle_rule(rem, battle, points, grouppoints):
  groups = jigsaw_to_coords(battle)
  l = len(rem)
  for (p, q) in frozenset.union(grouppoints, *(cell_visibility(i, j, l, (king_rule_points,)) for i, j in points)):
    if 1 in rem[p][q]:
      rem[p][q].remove(1)
  return rem
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
def get_squares(points):
  regions = []
  for i, j in points:
    if (i+1, j) in points and (i, j+1) in points and (i+1, j+1) in points:
      regions.append(frozenset(((i, j), (i+1, j), (i, j+1), (i+1, j+1))))
  return regions
def get_trominos(points):
  regions = []
  for i, j in points:
    if (i+1, j) in points and (i+1, j+1) in points and not (i, j+1) in points: #down, right
      regions.append(frozenset(((i, j), (i+1, j), (i+1, j+1))))
    if (i, j+1) in points and (i+1, j+1) in points and not (i+1, j) in points: #right, down
      regions.append(frozenset(((i, j), (i, j+1), (i+1, j+1))))
    if (i+1, j) in points and (i+1, j-1) in points and not (i, j-1) in points: #down, left
      regions.append(frozenset(((i, j), (i+1, j), (i+1, j-1))))
    if (i, j+1) in points and (i-1, j+1) in points and not (i-1, j) in points: #right, up
      regions.append(frozenset(((i, j), (i, j+1), (i-1, j+1))))
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
  regions = []
  for i in range(0, l, 2):
    for j in range(0, l, 2):
      regions.append(frozenset(((i, j), (i+1, j), (i, j+1), (i+1, j+1))))
  return regions
def get_mutex_regions(points, numstars):
  #need all solutions to equation 4s+3t+2d+p=len(points), s+t+d+p=numstars naive way is easiest, perhaps Diophantine equation can be solved more efficiently
  #perms = []
  #a domino excludes its 4 length-wise orthagonal squares, a diagonal domino excludes the alternate corners in its square region
  #a tromino excludes its inner point in its square region
  combs = []
  #if numstars * 4 < len(points): return frozenset()
  squares = get_squares(points)
  for s in range(numstars, -1, -1):
    for comb in itertools.combinations(squares, s):
      u = frozenset() if s == 0 else frozenset.union(*comb)
      if len(u) == s << 2:
        r = points - u
        trominos = get_trominos(r)
        for t in range(numstars - s, -1, -1):
          for combtrom in itertools.combinations(trominos, t):
            ut = frozenset() if t == 0 else frozenset.union(*combtrom)
            if len(ut) == t * 3:
              rt = r - ut
              dominos = get_dominos(rt)
              for d in range(numstars - s - t, -1, -1):
                for combdom in itertools.combinations(dominos, d):
                  ud = frozenset() if d == 0 else frozenset.union(*combdom)
                  if len(ud) == d << 1:
                    rd = rt - ud
                    diag_dominos = get_diag_dominos(rd)
                    for dg in range(numstars - s - t - d, -1, -1):
                      for combddom in itertools.combinations(diag_dominos, dg):
                        udd = frozenset() if dg == 0 else frozenset.union(*combddom)
                        if len(udd) == dg << 1:
                          if len(u) + len(ut) + len(ud) + len(udd) + numstars - s - t - d - dg == len(points):
                            rdd = rd - udd
                            if any(len(rd.intersection(king_rule_pts(pt[0], pt[1]))) != 0 for pt in rdd): continue
                            #print(len(comb), len(combtrom), len(combdom))
                            combs.append(comb + combtrom + combdom + combddom + tuple(frozenset((x,)) for x in rdd))
  return combs
def exclude_star_battle(rem, battle, cell_star_battle_remove):
  stars, l = 2, len(rem)
  ret = None
  groups = jigsaw_to_coords(battle)
  #central squares in region that needs 2 stars and can see all others is eliminated
  for points in groups.values():
    numstars = sum((1 in rem[i][j] and len(rem[i][j]) == 1 for i, j in points))
    if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
    rempoints = frozenset(filter(lambda p: len(rem[p[0]][p[1]]) != 1, points))
    for i, j in rempoints:
      altpoints = frozenset.union(*king_rule_points(i, j, l))
      if all((x, y) in altpoints for x, y in rempoints - frozenset(((i, j),))):
        rem, ret = cell_star_battle_remove(rem, i, j, 1), True
  regions = tuple(tuple(x) for x in standard_sudoku_singoverlap_regions(l))
  for p in range(2, l-1):
    for i in range(l-p+1):
      #number of bounded regions equal to number of rows/columns
      for region in regions:
        regionpoints = frozenset.union(*(region[i+q] for q in range(p)))
        tot = frozenset(tuple(battle[x][y] for x, y in regionpoints))
        contained = tuple(x for x in tot if all(p in regionpoints for p in groups[x]))
        if len(contained) == p:
          points = set.union(*(groups[x] for x in contained))
          for x, y in regionpoints - points:
            if 1 in rem[x][y]: rem, ret = cell_star_battle_remove(rem, x, y, 1), True
  for i in range(l):
    for j in range(l):
      if len(rem[i][j]) == 2: #equivalent of claiming locked candidates
        vispoints = frozenset.union(*king_rule_points(i, j, l))
        altregions = frozenset(tuple(battle[p[0]][p[1]] for p in vispoints)) - frozenset((battle[i][j],))
        for x in altregions:
          numstars = sum((1 in rem[i][j] and len(rem[i][j]) == 1 for i, j in groups[x]))
          if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
          pts = tuple(filter(lambda p: len(rem[p[0]][p[1]]) == 2, groups[x] - vispoints))
          if len(pts) == 1 or len(pts) == 2 and len(get_dominos(pts)) == 1:
            #print(i, j, x, tuple(len(rem[p[0]][p[1]]) for p in groups[x] - vispoints), groups[x] - vispoints)
            rem, ret = cell_star_battle_remove(rem, i, j, 1), True
            break
        for vispoints in (*row_col_points(i, j, l), *jigsaw_points_gen(groups.values())(i, j, l)):
          numstars = sum((1 in rem[i][j] and len(rem[i][j]) == 1 for i, j in vispoints))
          if numstars != 1: continue
          altregions = frozenset(tuple(battle[p[0]][p[1]] for p in vispoints)) - frozenset((battle[i][j],))
          for x in altregions:
            numstars = sum((1 in rem[i][j] and len(rem[i][j]) == 1 for i, j in groups[x]))
            if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
            pts = tuple(filter(lambda p: len(rem[p[0]][p[1]]) == 2, groups[x] - vispoints))
            if len(pts) == 1 or len(pts) == 2 and len(get_dominos(pts)) == 1:
              #print(i, j, x, tuple(len(rem[p[0]][p[1]]) for p in groups[x] - vispoints), groups[x] - vispoints)
              rem, ret = cell_star_battle_remove(rem, i, j, 1), True
              break
      elif len(rem[i][j]) == 1 and 1 in rem[i][j]:
        for vispoints in row_col_points(i, j, l):
          numstars = sum((1 in rem[i][j] and len(rem[i][j]) == 1 for i, j in vispoints))
          if numstars != 1: continue
          altregions = frozenset(tuple(battle[p[0]][p[1]] for p in vispoints)) - frozenset((battle[i][j],))
          for x in altregions:
            numstars = sum((1 in rem[i][j] and len(rem[i][j]) == 1 for i, j in groups[x]))
            if numstars == stars or numstars == stars - 1: continue #applies only if 2 or more stars remaining
            pts = tuple(filter(lambda p: len(rem[p[0]][p[1]]) == 2, groups[x] - vispoints))
            if len(pts) == 1:
              #print(i, j, pts, x, tuple(len(rem[p[0]][p[1]]) for p in groups[x] - vispoints), groups[x] - vispoints)
              rem, ret = cell_star_battle_remove(rem, pts[0][0], pts[0][1], 0), True
              break
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
              if not (x, y) in shape: rem, ret = cell_star_battle_remove(rem, x, y, 1), True
  for i in range(l-1):
    for j in range(l-1): #all combinations of connecting regions up to some length not just 2x2 ones should be considered
      tot = frozenset((battle[i][j], battle[i][j+1], battle[i+1][j], battle[i+1][j+1]))
      points = set.union(*(groups[x] for x in tot))
      squares = get_square_regions(points)
      #if not squares is None:
      #  if len(squares) != stars * len(tot):
      #    print(tot, squares, len(squares), len(points))
      curstars = sum(1 for x in points if len(rem[x[0]][x[1]]) == 1 and next(iter(rem[x[0]][x[1]])) == 1)
      numstars = stars * len(tot) - curstars
      points = {x for x in points if len(rem[x[0]][x[1]]) != 1}
      combs = get_mutex_regions(points, numstars)
      if len(combs) != 0: #point forcing
        p = frozenset(y for x in combs for y in x if len(y) == 1)
        for s in p:
          if all(s in q for q in combs):
            if 0 in rem[next(iter(s))[0]][next(iter(s))[1]]: rem, ret = cell_star_battle_remove(rem, next(iter(s))[0], next(iter(s))[1], 0), True
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
                    rem, ret = cell_star_battle_remove(rem, pts[0][0], x, 1), True
          else:
            starsonrowcol = tuple(x for x in range(l) if len(rem[x][pts[0][1]]) == 1 and 1 in rem[x][pts[0][1]])
            if len(starsonrowcol) == 1:
              for x in range(l):
                if x != starsonrowcol[0] and x != pts[0][0] and x != pts[1][0]:
                  if 1 in rem[x][pts[0][1]]:
                    rem, ret = cell_star_battle_remove(rem, x, pts[0][1], 1), True
        for shape in pd: #domino exclusions
          pts = tuple(shape)
          for x, y in frozenset.intersection(*(z for q in range(len(pts)) for z in king_rule_points(pts[q][0], pts[q][1], l))):
            if 1 in rem[x][y]:
              rem, ret = cell_star_battle_remove(rem, x, y, 1), True
        pt = frozenset(y for x in combs for y in x if len(y) == 3)
        for shape in pt: #tromino exclusions
          pts = tuple(shape)
          for x, y in frozenset.intersection(*(z for q in range(len(pts)) for z in king_rule_points(pts[q][0], pts[q][1], l))):
            if 1 in rem[x][y]:
              rem, ret = cell_star_battle_remove(rem, x, y, 1), True
  return rem, ret
def star_battle_loop(rem, battle):
  def cell_star_battle_remove(rem, i, j, y):
    if not y in rem[i][j]: return rem
    rem[i][j].remove(y)
    return check_star_battle_rule(rem, battle)
  solve_path = []
  while True:
    rem, found = exclude_star_battle(rem, battle, cell_star_battle_remove)
    if any(any(len(y) == 0 for y in x) for x in rem): return None
    if found is None: break
    solve_path.append(found)
  return rem
def solve_star_battle(battle, stars):
  l = len(battle)
  rem = [[set((0, 1)) for _ in range(l)] for _ in range(l)]
  #groups = jigsaw_to_coords(battle)
  rem = star_battle_loop(rem, battle)
  if rem is None: print("Bad Star Battle")
  return rem

def get_star_battles():
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
  return ((lm_star_battle, 2),
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
  rem, valid = valid_star_battle(solve_star_battle(battle[0], battle[1]), battle[0], battle[1])
  if not rem is None:
    print_border(rem, exc_from_border(battle[0], set()))
    if not valid:
      print_candidate_border(rem, exc_from_border(battle[0], set()))
def check_star_battles(battles):
  for y in battles: check_star_battle(y)

check_puzzles(get_ctc_puzzles())
#check_puzzles(get_impossible_puzzles())
#check_puzzles(file_puzzles("topn87.txt"))

check_star_battles(get_star_battles())
