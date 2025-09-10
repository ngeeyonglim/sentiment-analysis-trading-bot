# password include uppercase, lowercase, digit, special character, and length of 8 characters

def check_password(pw):
  upper_case = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
  lower_case = ['a', 'b', 'c', 'd', 'e', 'f']
  digits = ['1', '2', '3', '4', '5']
  special_characters = ['!', '$', '_']

  if len(pw) < 8:
    return False

  upc = False
  for i in upper_case:
    if i in pw:
      upc = True
      break
    
  lwc = False
  for i in lower_case:
    if i in pw:
      lwc = True
      break
  
  dig = False
  for i in digits:
    if i in pw:
      dig = True
      break
  
  spec = False
  for i in special_characters:
    if i in pw:
      spec = True
      break
  
  return upc and lwc and dig and spec


pw1 = ['AB1ab!', 'AAAAAaaa', 'Ab1!$_cde'] # < 8 characters, no special characters, valid test case
test_results = [False, False, True]

def test(pwlist, test_results):
  res = []
  for i in pwlist:
    res.append(check_password(i))
  return res == test_results

print(test(pw1, test_results))
