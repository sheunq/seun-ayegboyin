
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, BotMessageSquare, Newspaper, Home, Github, Linkedin } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { cn } from '@/lib/utils';
import { useState } from 'react';
import { ModeToggle } from './ModeToggle';

const navLinks = [
  { href: '/', label: 'Home', icon: Home },
  { href: '/blog', label: 'Blog', icon: Newspaper },
  { href: '/summarizer', label: 'AI Summarizer', icon: BotMessageSquare },
];

const WhatsAppIcon = (props: React.SVGProps<SVGSVGElement>) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
    </svg>
  );

export default function Header() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  const NavLink = ({ href, label, icon: Icon }: typeof navLinks[0]) => (
    <Link
      href={href}
      onClick={() => setIsOpen(false)}
      className={cn(
        'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
        pathname === href
          ? 'bg-accent text-accent-foreground'
          : 'text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground'
      )}
    >
      <Icon className="h-4 w-4" />
      <span>{label}</span>
    </Link>
  );

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 max-w-screen-2xl items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <BotMessageSquare className="h-6 w-6 text-primary" />
          <span className="font-headline text-lg font-bold">Seun Ayegboyin</span>
        </Link>

        <div className="flex items-center gap-2">
            <nav className="hidden items-center gap-2 md:flex">
            {navLinks.map((link) => (
                <Link
                href={link.href}
                key={link.href}
                className={cn(
                    'rounded-md px-3 py-2 text-sm font-medium transition-colors',
                    pathname === link.href
                    ? 'bg-primary/10 text-primary'
                    : 'text-foreground/60 hover:text-foreground/80'
                )}
                >
                {link.label}
                </Link>
            ))}
            </nav>
            <div className="hidden md:flex items-center gap-2">
                <Button variant="ghost" size="icon" asChild>
                  <Link href="https://github.com/sheunq" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                    <Github className="h-5 w-5" />
                  </Link>
                </Button>
                <Button variant="ghost" size="icon" asChild>
                  <Link href="https://www.linkedin.com/in/seun-ayegboyin-36a74a176" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                    <Linkedin className="h-5 w-5" />
                  </Link>
                </Button>
                <Button variant="ghost" size="icon" asChild>
                  <Link href="https://wa.me/2347066885273" target="_blank" rel="noopener noreferrer" aria-label="WhatsApp">
                    <WhatsAppIcon className="h-5 w-5" />
                  </Link>
                </Button>
            </div>
            <ModeToggle />

            <div className="md:hidden">
                <Sheet open={isOpen} onOpenChange={setIsOpen}>
                    <SheetTrigger asChild>
                    <Button variant="outline" size="icon">
                        <Menu className="h-4 w-4" />
                        <span className="sr-only">Open menu</span>
                    </Button>
                    </SheetTrigger>
                    <SheetContent side="right">
                    <div className="flex flex-col gap-4 p-4">
                        <Link href="/" className="mb-4 flex items-center gap-2">
                        <BotMessageSquare className="h-6 w-6 text-primary" />
                        <span className="font-headline text-lg font-bold">Seun Ayegboyin</span>
                        </Link>
                        <div className="flex flex-col gap-2">
                        {navLinks.map((link) => (
                            <NavLink key={link.href} {...link} />
                        ))}
                        </div>
                        <div className="mt-4 flex gap-2">
                            <Button variant="ghost" size="icon" asChild>
                              <Link href="https://github.com/sheunq" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                                <Github className="h-5 w-5" />
                              </Link>
                            </Button>
                            <Button variant="ghost" size="icon" asChild>
                              <Link href="https://www.linkedin.com/in/seun-ayegboyin-36a74a176" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                                <Linkedin className="h-5 w-5" />
                              </Link>
                            </Button>
                            <Button variant="ghost" size="icon" asChild>
                                <Link href="https://wa.me/2347066885273" target="_blank" rel="noopener noreferrer" aria-label="WhatsApp">
                                    <WhatsAppIcon className="h-5 w-5" />
                                </Link>
                            </Button>
                        </div>
                    </div>
                    </SheetContent>
                </Sheet>
            </div>
        </div>
      </div>
    </header>
  );
}
